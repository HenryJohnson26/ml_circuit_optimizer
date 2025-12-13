# cnn.py
"""
Updated CNN training script that:
 - Predicts full tokens (gate_id, q1, q2)
 - Uses a differentiable PyTorch statevector simulator for a fidelity penalty
 - Keeps your structural penalty and training loop style
 - Infers n_qubits from data; maps q2=-1 -> NO_Q_IDX
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data/tokenized"
WINDOW = 32
BATCH_SIZE = 32
EPOCHS = 10
EMBED_DIM = 32
Q_EMBED = 4
HIDDEN_DIM = 128
LR = 1e-3
TRAINING_SPLIT = 0.8

if torch.cuda.is_available():
    print("Using gpu")
    DEVICE = torch.device("cuda")
else:
    print("Using cpu")
    DEVICE = torch.device("cpu")

# gates in your vocab (kept here for readability)
GATE_VOCAB = [
    "h", "s", "sdg", "x", "y", "z", "t", "tdg", "rz_pi_4",
    "cx", "cz"
]

# token indices of non-Cliffords (keep same as you used; adjust if needed)
NONCLIFFORD_GATES = [2, 7, 8, 9, 10]  # NOTE: verify these indices map to your vocab if different

# Fidelity penalty config (differentiable)
LAMBDA_FID = 1.0
SAMPLE_FRACTION_FOR_FID = 0.25  # fraction of batch to compute fidelity on (keeps compute reasonable)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------
# Utility: infer n_qubits from data
# ----------------------------
def load_all_jsons(data_dir: str):
    raw_sequences = []
    opt_sequences = []
    jsons = list(Path(data_dir).glob("*.json"))
    for f in jsons:
        with open(f, "r") as fp:
            d = json.load(fp)
            raw_sequences.append(d["raw_ints"])
            opt_sequences.append(d["opt_ints"])
    return raw_sequences, opt_sequences, jsons


raw_sequences, opt_sequences, json_files = load_all_jsons(DATA_DIR)
print(f"Loaded {len(raw_sequences)} circuits from {DATA_DIR}")

# split
n_total = len(raw_sequences)
split_idx = int(n_total * TRAINING_SPLIT)
raw_training_sequences = raw_sequences[:split_idx]
raw_val_sequences = raw_sequences[split_idx:]
opt_training_sequences = opt_sequences[:split_idx]
opt_val_sequences = opt_sequences[split_idx:]


# ----------------------------
# Build dataset tensors (X windows and Y full tokens)
# ----------------------------
# infer maximum qubit index across datasets
max_q = -1
for f in json_files:
    with open(f, "r") as fp:
        d = json.load(fp)
        all_tokens = d.get("raw_tokens", []) + d.get("opt_tokens", [])

        for token in all_tokens:
            # token like ["cx", [0, 1]] OR ["h", [0]]
            if len(token) > 1 and isinstance(token[1], list):
                # check each value inside the argument list
                for arg in token[1]:
                    if isinstance(arg, int) and arg >= 0:
                        max_q = max(max_q, arg)
            # some formats may put args inside token[2] as well
            if len(token) > 2 and isinstance(token[2], list):
                for arg in token[2]:
                    if isinstance(arg, int) and arg >= 0:
                        max_q = max(max_q, arg)

if max_q < 0:
    raise RuntimeError("No valid qubit indices found in token data")

NUM_QUBITS = max_q + 1
NO_Q_IDX = NUM_QUBITS

print(f"Corrected NUM_QUBITS = {NUM_QUBITS}, NO_Q_IDX = {NO_Q_IDX}")

# helper to build X,Y lists
def build_dataset_lists(raw_seqs, opt_seqs, window=WINDOW):
    X_list = []
    Y_list = []  # will store [gate_id, q1, q2_mapped]
    for raw, opt in zip(raw_seqs, opt_seqs):
        length = min(len(raw), len(opt))
        for i in range(0, length - window - 1):
            win = raw[i:i + window]
            nxt = opt[i + window]
            # win is list of [gate_id, q1, q2]
            # map q2=-1 -> NO_Q_IDX
            g_id, q1, q2 = map(int, nxt)
            q2_mapped = q2 if q2 >= 0 else NO_Q_IDX
            X_list.append(win)
            Y_list.append([g_id, q1, q2_mapped])
    return X_list, Y_list

X_train_list, Y_train_list = build_dataset_lists(raw_training_sequences, opt_training_sequences, WINDOW)
X_val_list, Y_val_list = build_dataset_lists(raw_val_sequences, opt_val_sequences, WINDOW)

print(f"Train samples: {len(X_train_list)}  Val samples: {len(X_val_list)}")

# Convert to tensors
if len(X_train_list) == 0:
    raise RuntimeError("No training samples found. Check WINDOW / dataset length.")

X_train = torch.tensor(X_train_list, dtype=torch.long)
Y_train = torch.tensor(Y_train_list, dtype=torch.long)

X_val = torch.tensor(X_val_list, dtype=torch.long) if len(X_val_list) > 0 else torch.empty((0, WINDOW, 3), dtype=torch.long)
Y_val = torch.tensor(Y_val_list, dtype=torch.long) if len(Y_val_list) > 0 else torch.empty((0, 3), dtype=torch.long)

print("Shapes: X_train:", X_train.shape, "Y_train:", Y_train.shape)


train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------
# Model
# ----------------------------
# compute gate vocab from data if possible
gate_ids = X_train[:, :, 0]
VOCAB_SIZE = int(gate_ids.max().item() + 1)
print("Inferred VOCAB_SIZE:", VOCAB_SIZE)

class CNNPredictor(nn.Module):
    def __init__(self, gate_vocab: int, num_qubits: int, embed_dim: int = EMBED_DIM, q_embed: int = Q_EMBED, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.g_emb = nn.Embedding(gate_vocab, embed_dim)
        self.q1_emb = nn.Embedding(num_qubits, q_embed)
        # q2 includes NO_Q_IDX so we provide num_qubits + 1 classes externally, but keep embedding size same
        self.q2_emb = nn.Embedding(num_qubits + 1, q_embed)
        self.pos_emb = nn.Embedding(WINDOW, 8)

        total_ch = embed_dim + q_embed + q_embed + 8
        self.conv = nn.Conv1d(total_ch, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # heads: gate, q1, q2
        self.fc_gate = nn.Linear(hidden_dim * WINDOW, gate_vocab)
        self.fc_q1 = nn.Linear(hidden_dim * WINDOW, num_qubits)         # q1 must be in [0, num_qubits-1]
        self.fc_q2 = nn.Linear(hidden_dim * WINDOW, num_qubits + 1)     # q2 can be [0..num_qubits-1] or NO_Q_IDX

    def forward(self, x):
        # x: (B, W, 3)
        gate_ids = x[:, :, 0]
        q1_ids = x[:, :, 1]
        q2_ids = x[:, :, 2]  # already mapped with NO_Q_IDX in dataset

        g_e = self.g_emb(gate_ids)
        q1_e = self.q1_emb(q1_ids)
        q2_e = self.q2_emb(q2_ids)
        positions = torch.arange(WINDOW, device=x.device)
        pos_e = self.pos_emb(positions)[None, :, :].expand(x.size(0), -1, -1)

        cat = torch.cat([g_e, q1_e, q2_e, pos_e], dim=-1)  # (B, W, total_ch)
        cat = cat.permute(0, 2, 1)  # (B, C, L)
        h = self.relu(self.conv(cat))  # (B, hidden_dim, L)
        h_flat = h.flatten(1)  # (B, hidden_dim*L)

        gate_logits = self.fc_gate(h_flat)
        q1_logits = self.fc_q1(h_flat)
        q2_logits = self.fc_q2(h_flat)
        return gate_logits, q1_logits, q2_logits


model = CNNPredictor(gate_vocab=VOCAB_SIZE, num_qubits=NUM_QUBITS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
ce_loss = nn.CrossEntropyLoss()


# ----------------------------
# Differentiable PyTorch simulator helpers
# ----------------------------
# We construct single-qubit matrices as complex128 torch tensors on DEVICE.
complex_dtype = torch.cfloat if DEVICE.type == "cpu" else torch.complex128  # prefer 128 on GPU if available
# Note: torch.complex128 may be slower on some GPUs; adjust if necessary.

def get_single_qubit_matrices(device):
    dtype = torch.complex128 if torch.complex128 in (torch.tensor(0).to(device).dtype, torch.complex128) else torch.cfloat
    # build matrices explicitly as complex tensors
    I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    H = (1.0 / np.sqrt(2.0)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128, device=device)
    S = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex128, device=device)
    Sdg = S.conj().permute(1, 0)
    T = torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=torch.complex128, device=device)
    Tdg = T.conj().permute(1, 0)
    def Rz(angle):
        a = float(angle)
        return torch.tensor([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]], dtype=torch.complex128, device=device)
    return {
        "I": I, "X": X, "Y": Y, "Z": Z, "H": H,
        "S": S, "SDG": Sdg, "T": T, "TDG": Tdg, "RZ": Rz
    }

# build mapping from gate_id -> gate_name (if you need to print later)
GATE_ID_TO_NAME = {i: name for i, name in enumerate(GATE_VOCAB)}

# Precompute single-qubit mats dictionary on DEVICE
_SQ_MATS = get_single_qubit_matrices(DEVICE)

def embed_single_unitary(n_qubits: int, single_mat: torch.Tensor, target: int) -> torch.Tensor:
    # build full unitary by kronecker product; qubit 0 is LSB so we reverse order
    mats = []
    for i in range(n_qubits):
        if i == target:
            mats.append(single_mat)
        else:
            mats.append(_SQ_MATS["I"])
    # reverse to make qubit0 LSB -> rightmost in kronecker
    mats = list(reversed(mats))
    U = mats[0]
    for m in mats[1:]:
        U = torch.kron(U, m)
    return U  # shape (2^n, 2^n) complex

def cx_full_torch(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    dim = 2 ** n_qubits
    # build as dense matrix using basis mapping
    U = torch.eye(dim, dtype=torch.complex128, device=device)
    for basis in range(dim):
        if ((basis >> control) & 1) == 1:
            flipped = basis ^ (1 << target)
            # set column 'basis' to basis->flipped
            U[flipped, basis] = 1.0 + 0j
            U[basis, basis] = 0.0 + 0j
    return U

def cz_full_torch(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    dim = 2 ** n_qubits
    U = torch.eye(dim, dtype=torch.complex128, device=device)
    for basis in range(dim):
        if (((basis >> control) & 1) == 1) and (((basis >> target) & 1) == 1):
            U[basis, basis] *= -1.0 + 0j
    return U

# Convenience to build full unitary for a token (gate_name string or gate_id)
def build_full_unitary_by_name(gate_name: str, q1: int, q2: Optional[int], n_qubits: int, angle: Optional[float], device: torch.device) -> torch.Tensor:
    # gate_name examples: 'h','s','sdg','x',... 'cx','cz','rz_pi_4'
    # treat 'rz' family via angle if provided
    if gate_name.startswith("rz"):
        ang = (angle if angle is not None else (np.pi / 4))
        return embed_single_unitary(n_qubits, _SQ_MATS["RZ"](ang), q1)
    if gate_name == "h":
        return embed_single_unitary(n_qubits, _SQ_MATS["H"], q1)
    if gate_name == "s":
        return embed_single_unitary(n_qubits, _SQ_MATS["S"], q1)
    if gate_name == "sdg":
        return embed_single_unitary(n_qubits, _SQ_MATS["SDG"], q1)
    if gate_name == "x":
        return embed_single_unitary(n_qubits, _SQ_MATS["X"], q1)
    if gate_name == "y":
        return embed_single_unitary(n_qubits, _SQ_MATS["Y"], q1)
    if gate_name == "z":
        return embed_single_unitary(n_qubits, _SQ_MATS["Z"], q1)
    if gate_name == "t":
        return embed_single_unitary(n_qubits, _SQ_MATS["T"], q1)
    if gate_name == "tdg":
        return embed_single_unitary(n_qubits, _SQ_MATS["TDG"], q1)
    if gate_name == "cx":
        return cx_full_torch(n_qubits, q1, q2, device)
    if gate_name == "cz":
        return cz_full_torch(n_qubits, q1, q2, device)
    raise NotImplementedError(f"Gate name {gate_name} not in unitary builder")

# Simulate a sequence of rich tokens (gate_name, q1, q2, angle) using torch
def simulate_sequence_torch(rich_tokens: List[Tuple[str, int, Optional[int], Optional[float]]], n_qubits: int, device: torch.device) -> torch.Tensor:
    dim = 2 ** n_qubits
    state = torch.zeros(dim, dtype=torch.complex128, device=device)
    state[0] = 1.0 + 0j
    for gate_name, q1, q2, angle in rich_tokens:
        U = build_full_unitary_by_name(gate_name, q1, q2, n_qubits, angle, device)
        state = U @ state
    return state  # complex vector length 2^n

# ----------------------------
# Differentiable fidelity penalty
# ----------------------------
def fidelity_penalty_differentiable(batch_x: torch.Tensor, batch_y: torch.Tensor, model: nn.Module, sample_frac: float = SAMPLE_FRACTION_FOR_FID) -> torch.Tensor:
    """
    batch_x: (B, W, 3) ints (gate_id, q1, q2_mapped)
    batch_y: (B, 3) ints (gate_id, q1, q2_mapped)
    We will:
      - For each sampled example, simulate state after window (using token ints -> gate names)
      - Compute label state = U_label |psi>
      - Compute predicted state = sum_{g,q1,q2} p_g * p_q1 * p_q2 * U(g,q1,q2) |psi>
      - fidelity = |<label|pred>|^2
      - penalty = mean(1 - fidelity)
    Returns a scalar torch Tensor (differentiable wrt model's logits via softmax probs).
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # but we still need gradients for the model outputs; we'll recompute logits below without torch.no_grad()
        pass

    # compute logits WITHOUT torch.no_grad() to allow gradients
    gate_logits, q1_logits, q2_logits = model(batch_x.to(device))
    # convert to probabilities with softmax (differentiable)
    p_gates = torch.softmax(gate_logits, dim=-1)       # (B, G)
    p_q1 = torch.softmax(q1_logits, dim=-1)            # (B, Q)
    p_q2 = torch.softmax(q2_logits, dim=-1)            # (B, Q+1)

    B = batch_x.size(0)
    n_sample = max(1, int(B * sample_frac))
    indices = random.sample(range(B), n_sample)

    penalties = []
    for i in indices:
        # reconstruct window -> rich token sequence (we only have ints)
        win = batch_x[i].cpu().numpy().tolist()  # list of [g_id, q1, q2_mapped]
        seq_rich = []
        for (g_id, q1, q2_m) in win:
            g_name = GATE_VOCAB[int(g_id)]
            q1_i = int(q1)
            q2_i = (int(q2_m) if int(q2_m) != NO_Q_IDX else None)
            # no angle info available; use default angle for rz_pi_4
            angle = (np.pi / 4) if g_name.startswith("rz") else None
            seq_rich.append((g_name, q1_i, q2_i, angle))

        # initial state after window (tensor on device)
        state_before = simulate_sequence_torch(seq_rich, NUM_QUBITS, device)  # complex tensor

        # Ground truth next token (use batch_y)
        gy, gq1, gq2_m = batch_y[i].cpu().numpy().tolist()
        gt_gate_name = GATE_VOCAB[int(gy)]
        gt_q1 = int(gq1)
        gt_q2 = (int(gq2_m) if int(gq2_m) != NO_Q_IDX else None)
        gt_angle = (np.pi / 4) if gt_gate_name.startswith("rz") else None
        # label state
        label_tok = [(gt_gate_name, gt_q1, gt_q2, gt_angle)]
        state_label = simulate_sequence_torch(label_tok, NUM_QUBITS, device)  # this applies *only* the next gate; we need U_label|psi_before
        # need to multiply: compute U_label @ state_before
        U_label = build_full_unitary_by_name(gt_gate_name, gt_q1, gt_q2, NUM_QUBITS, gt_angle, device)
        state_label_full = U_label @ state_before  # complex vector

        # Predicted state: expected over predicted distributions p(g)*p(q1)*p(q2)
        # factorized joint -> iterate over gates and q1,q2 supports (exclude NO_Q for 2q gates)
        p_g = p_gates[i]   # (G,)
        p_q1_row = p_q1[i] # (Q,)
        p_q2_row = p_q2[i] # (Q+1,)

        # Build predicted state as weighted sum of U(g,q1,q2) @ state_before
        state_pred = torch.zeros_like(state_label_full, device=device, dtype=torch.complex128)

        # iterate over gates
        G = p_g.size(0)
        Q = p_q1_row.size(0)
        Q2 = p_q2_row.size(0)
        for g_idx in range(G):
            pg = p_g[g_idx]  # scalar prob
            gate_name = GATE_VOCAB[g_idx]
            if gate_name in ["cx", "cz"]:
                # two-qubit gate: iterate q1 in 0..Q-1, q2 in 0..Q-1 (exclude NO_Q_IDX)
                for q1_idx in range(Q):
                    pq1 = p_q1_row[q1_idx]
                    for q2_idx in range(Q):
                        # q2_idx maps to 0..Q-1; we do not include NO_Q_IDX here (which is index Q)
                        pq2 = p_q2_row[q2_idx]
                        joint_p = pg * pq1 * pq2
                        if joint_p.item() == 0.0:
                            continue
                        U = build_full_unitary_by_name(gate_name, q1_idx, q2_idx, NUM_QUBITS, None, device)
                        state_pred = state_pred + joint_p * (U @ state_before)
            else:
                # single-qubit gate: q2 is irrelevant; iterate q1 over 0..Q-1
                for q1_idx in range(Q):
                    pq1 = p_q1_row[q1_idx]
                    # we can sum over q2 probabilities as well but it's redundant (same U) so we sum pq2 total to multiply joint
                    pq2_total = p_q2_row.sum()  # should be 1.0 but includes NO_Q class — fine
                    joint_p = pg * pq1 * pq2_total
                    if joint_p.item() == 0.0:
                        continue
                    U = build_full_unitary_by_name(gate_name, q1_idx, None, NUM_QUBITS, None, device)
                    state_pred = state_pred + joint_p * (U @ state_before)

        # compute fidelity between state_label_full and state_pred
        overlap = torch.vdot(state_label_full, state_pred)
        fidelity = torch.abs(overlap) ** 2
        penalty = 1.0 - fidelity
        penalties.append(penalty)

    if len(penalties) == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    penalties_tensor = torch.stack(penalties)  # (n_sample,)
    return penalties_tensor.mean()  # differentiable scalar


# ----------------------------
# Structural penalty (updated to accept gate logits)
# ----------------------------
def structural_penalty(gate_logits: torch.Tensor) -> torch.Tensor:
    """
    gate_logits: (B, G) logits over gate vocabulary for the next prediction
    Penalty: encourage lower probability mass on non-Clifford gates early in the window.
    We apply structural penalty only to gate logits (not q args).
    """
    probs = torch.softmax(gate_logits, dim=-1)  # (B, G)
    nonc_mask = torch.zeros(probs.size(1), device=probs.device)
    for idx in NONCLIFFORD_GATES:
        if idx < probs.size(1):
            nonc_mask[idx] = 1.0
    nonc_prob = (probs * nonc_mask[None, :]).sum(dim=-1)  # (B,)
    # We want to weight earlier positions? But gate_logits are for next token, so a simple mean is fine
    return nonc_prob.mean()


# ----------------------------
# Eval helper: non-clifford clusters (keeps older behavior)
# ----------------------------
def non_clifford_clusters(seq):
    clusters = 0
    in_cluster = False
    for tok in seq:
        if tok in NONCLIFFORD_GATES:
            if not in_cluster:
                clusters += 1
            in_cluster = True
        else:
            in_cluster = False
    return clusters

def eval_clusters(model, dataset_loader, n=20):
    model.eval()
    before_total, after_total = 0, 0
    # dataset_loader.dataset is TensorDataset(X,Y)
    ds = dataset_loader.dataset
    N = len(ds)
    if N == 0:
        return
    indices = random.sample(range(N), min(n, N))
    for idx in indices:
        x, y = ds[idx]
        raw = x  # shape (W,3)
        raw_list = raw[:, 0].tolist()
        before = non_clifford_clusters(raw_list)
        # predict gate id
        with torch.no_grad():
            g_logits, _, _ = model(raw.unsqueeze(0).to(DEVICE))
            pred_gate = int(torch.argmax(g_logits, dim=-1).cpu().item())
        after = non_clifford_clusters([pred_gate])
        before_total += before
        after_total += after
    print(f"Avg clusters before: {before_total/len(indices):.2f} → after: {after_total/len(indices):.2f}")


# ----------------------------
# Training loop
# ----------------------------
def train():
    print("Starting training on device:", DEVICE)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            gate_logits, q1_logits, q2_logits = model(batch_x)

            # compute supervised CE losses
            gate_target = batch_y[:, 0]
            q1_target = batch_y[:, 1]
            q2_target = batch_y[:, 2]

            loss_gate = ce_loss(gate_logits, gate_target)
            loss_q1 = ce_loss(q1_logits, q1_target)
            loss_q2 = ce_loss(q2_logits, q2_target)
            loss = loss_gate + loss_q1 + loss_q2

            # add structural penalty (applied to gate logits)
            loss = loss + 0.3 * structural_penalty(gate_logits)

            # differentiable fidelity penalty (sample subset to save time)
            fid_pen = fidelity_penalty_differentiable(batch_x, batch_y, model, sample_frac=SAMPLE_FRACTION_FOR_FID)
            loss = loss + LAMBDA_FID * fid_pen

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.6f} fid_pen(last batch)={float(fid_pen.detach().cpu().numpy()):.6f}")
        eval_clusters(model, train_loader)
    print("Training finished.")


# ----------------------------
# Test / evaluation code (similar to yours but for full token preds)
# ----------------------------
def test_model():
    # build test lists similarly to earlier test code
    x_list = []
    y_list = []
    for raw, opt in zip(raw_val_sequences, opt_val_sequences):
        length = min(len(raw), len(opt))
        for i in range(0, length - WINDOW - 1):
            win = raw[i:i + WINDOW]
            nxt = opt[i + WINDOW]
            g_id, q1, q2 = map(int, nxt)
            q2_mapped = q2 if q2 >= 0 else NO_Q_IDX
            x_list.append(win)
            y_list.append([g_id, q1, q2_mapped])
    if len(x_list) == 0:
        print("No test samples available.")
        return
    X_test = torch.tensor(x_list, dtype=torch.long)
    Y_test = torch.tensor(y_list, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE)
    model.eval()
    total_loss = 0.0
    total_correct_gates = 0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            g_logits, q1_logits, q2_logits = model(xb)
            loss_g = ce_loss(g_logits, yb[:, 0])
            loss_q1 = ce_loss(q1_logits, yb[:, 1])
            loss_q2 = ce_loss(q2_logits, yb[:, 2])
            total_loss += float((loss_g + loss_q1 + loss_q2).detach().cpu().numpy()) * xb.size(0)

            _, pred_g = torch.max(g_logits, dim=-1)
            total_correct_gates += (pred_g.cpu() == yb[:, 0].cpu()).sum().item()
            total_samples += xb.size(0)
    avg_loss = total_loss / total_samples
    gate_accuracy = total_correct_gates / total_samples * 100.0
    print(f"Test avg_loss={avg_loss:.6f}, gate_acc={gate_accuracy:.2f}%")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    train()
    test_model()
