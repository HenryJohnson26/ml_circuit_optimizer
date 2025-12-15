# cnn.py
"""
Fixed CNN training script that:
 - Predicts full tokens (gate_id, q1, q2)
 - Uses a differentiable PyTorch statevector simulator for a fidelity penalty
 - Keeps structural penalty and training loop style
 - Infers n_qubits from data; maps q2=-1 -> NO_Q_IDX in BOTH X and Y
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
WINDOW = 16  # Reduced from 32 - shorter context is easier to learn
BATCH_SIZE = 64  # Increased for more stable gradients
EPOCHS = 50  # More epochs to learn
EMBED_DIM = 64  # Increased capacity
Q_EMBED = 16
HIDDEN_DIM = 256  # Increased capacity
LR = 5e-4  # Reduced learning rate
TRAINING_SPLIT = 0.8

if torch.cuda.is_available():
    print("Using gpu")
    DEVICE = torch.device("cuda")
else:
    print("Using cpu")
    DEVICE = torch.device("cpu")

# gates in your vocab
GATE_VOCAB = [
    "h", "s", "sdg", "x", "y", "z", "t", "tdg", "rz_pi_4",
    "cx", "cz"
]

# token indices of non-Cliffords (t=6, tdg=7, rz_pi_4=8)
# Note: S and Sdg ARE Clifford gates
NONCLIFFORD_GATES = [6, 7, 8]  # t, tdg, rz_pi_4

# Fidelity penalty config (differentiable)
LAMBDA_FID = 0.5  # Increased to make fidelity more important
SAMPLE_FRACTION_FOR_FID = 0.2  # Sample more for better gradient signal

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
        # Use raw_ints and opt_ints directly
        all_ints = d.get("raw_ints", []) + d.get("opt_ints", [])
        for token in all_ints:
            # token is [gate_id, q1, q2]
            if len(token) >= 2:
                q1 = token[1]
                if isinstance(q1, int) and q1 >= 0:
                    max_q = max(max_q, q1)
            if len(token) >= 3:
                q2 = token[2]
                if isinstance(q2, int) and q2 >= 0:
                    max_q = max(max_q, q2)

if max_q < 0:
    raise RuntimeError("No valid qubit indices found in token data")

NUM_QUBITS = max_q + 1
NO_Q_IDX = NUM_QUBITS

print(f"NUM_QUBITS = {NUM_QUBITS}, NO_Q_IDX = {NO_Q_IDX}")

# helper to build X,Y lists with proper q2 mapping
def build_dataset_lists(raw_seqs, opt_seqs, window=WINDOW):
    X_list = []
    Y_list = []  # will store [gate_id, q1, q2_mapped]
    for raw, opt in zip(raw_seqs, opt_seqs):
        length = min(len(raw), len(opt))
        if length <= window:
            continue  # Skip sequences that are too short
        for i in range(0, length - window):
            win = raw[i:i + window]
            nxt = opt[i + window]

            # Map q2=-1 to NO_Q_IDX for the window tokens
            win_mapped = []
            for tok in win:
                g_id, q1, q2 = map(int, tok)
                q2_mapped = q2 if q2 >= 0 else NO_Q_IDX
                win_mapped.append([g_id, q1, q2_mapped])

            # Map q2=-1 to NO_Q_IDX for the target token
            g_id, q1, q2 = map(int, nxt)
            q2_mapped = q2 if q2 >= 0 else NO_Q_IDX

            X_list.append(win_mapped)
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

# Verify no negative values in q2
assert (X_train[:, :, 2] >= 0).all(), "Found negative q2 values in X_train!"
assert (Y_train[:, 2] >= 0).all(), "Found negative q2 values in Y_train!"
print("✓ All q2 values properly mapped")

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
        self.q2_emb = nn.Embedding(num_qubits + 1, q_embed)
        self.pos_emb = nn.Embedding(WINDOW, 8)

        total_ch = embed_dim + q_embed + q_embed + 8

        # Deeper CNN with multiple layers
        self.conv1 = nn.Conv1d(total_ch, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # heads: gate, q1, q2
        self.fc_gate = nn.Sequential(
            nn.Linear(hidden_dim * WINDOW, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, gate_vocab)
        )
        self.fc_q1 = nn.Sequential(
            nn.Linear(hidden_dim * WINDOW, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_qubits)
        )
        self.fc_q2 = nn.Sequential(
            nn.Linear(hidden_dim * WINDOW, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_qubits + 1)
        )

    def forward(self, x):
        # x: (B, W, 3)
        gate_ids = x[:, :, 0]
        q1_ids = x[:, :, 1]
        q2_ids = x[:, :, 2]

        g_e = self.g_emb(gate_ids)
        q1_e = self.q1_emb(q1_ids)
        q2_e = self.q2_emb(q2_ids)
        positions = torch.arange(WINDOW, device=x.device)
        pos_e = self.pos_emb(positions)[None, :, :].expand(x.size(0), -1, -1)

        cat = torch.cat([g_e, q1_e, q2_e, pos_e], dim=-1)  # (B, W, total_ch)
        cat = cat.permute(0, 2, 1)  # (B, C, L)

        # Multi-layer CNN
        h = self.relu(self.bn1(self.conv1(cat)))
        h = self.dropout(h)
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.dropout(h)
        h = self.relu(self.bn4(self.conv4(h)))
        h = self.dropout(h)
        h = self.relu(self.bn5(self.conv5(h)))


        h = self.dropout(h)

        h_flat = h.flatten(1)  # (B, hidden_dim*L)

        gate_logits = self.fc_gate(h_flat)
        q1_logits = self.fc_q1(h_flat)
        q2_logits = self.fc_q2(h_flat)
        return gate_logits, q1_logits, q2_logits


model = CNNPredictor(gate_vocab=VOCAB_SIZE, num_qubits=NUM_QUBITS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # AdamW with weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
ce_loss = nn.CrossEntropyLoss()


# ----------------------------
# Differentiable PyTorch simulator helpers
# ----------------------------
def get_single_qubit_matrices(device):
    dtype = torch.complex64  # Use complex64 for better compatibility
    I = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    H = (1.0 / np.sqrt(2.0)) * torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device)
    S = torch.tensor([[1, 0], [0, 1j]], dtype=dtype, device=device)
    Sdg = S.conj().T
    T = torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=dtype, device=device)
    Tdg = T.conj().T
    def Rz(angle):
        a = float(angle)
        return torch.tensor([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]], dtype=dtype, device=device)
    return {
        "I": I, "X": X, "Y": Y, "Z": Z, "H": H,
        "S": S, "SDG": Sdg, "T": T, "TDG": Tdg, "RZ": Rz
    }

# Precompute single-qubit mats dictionary on DEVICE
_SQ_MATS = get_single_qubit_matrices(DEVICE)

def embed_single_unitary(n_qubits: int, single_mat: torch.Tensor, target: int) -> torch.Tensor:
    """Embed single-qubit unitary into n-qubit space"""
    mats = []
    for i in range(n_qubits):
        if i == target:
            mats.append(single_mat.clone())
        else:
            mats.append(_SQ_MATS["I"].clone())
    # Qubit 0 is LSB - rightmost in kronecker
    mats = list(reversed(mats))
    U = mats[0].contiguous()
    for m in mats[1:]:
        U = torch.kron(U, m.contiguous()).contiguous()
    return U

def cx_full_torch(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    """Build CX gate matrix"""
    dim = 2 ** n_qubits
    dtype = torch.complex64
    U = torch.eye(dim, dtype=dtype, device=device).contiguous()
    for basis in range(dim):
        if ((basis >> control) & 1) == 1:
            flipped = basis ^ (1 << target)
            U[:, basis] = 0
            U[flipped, basis] = 1.0 + 0j
    return U.contiguous()

def cz_full_torch(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    """Build CZ gate matrix"""
    dim = 2 ** n_qubits
    dtype = torch.complex64
    U = torch.eye(dim, dtype=dtype, device=device).contiguous()
    for basis in range(dim):
        if (((basis >> control) & 1) == 1) and (((basis >> target) & 1) == 1):
            U[basis, basis] = -1.0 + 0j
    return U.contiguous()

def build_full_unitary_by_name(gate_name: str, q1: int, q2: Optional[int], n_qubits: int, angle: Optional[float], device: torch.device) -> torch.Tensor:
    """Build full unitary for a gate"""
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
        if q2 is None:
            raise ValueError("CX requires q2")
        return cx_full_torch(n_qubits, q1, q2, device)
    if gate_name == "cz":
        if q2 is None:
            raise ValueError("CZ requires q2")
        return cz_full_torch(n_qubits, q1, q2, device)
    raise NotImplementedError(f"Gate name {gate_name} not in unitary builder")

def simulate_sequence_torch(rich_tokens: List[Tuple[str, int, Optional[int], Optional[float]]], n_qubits: int, device: torch.device) -> torch.Tensor:
    """Simulate a sequence of gates"""
    dim = 2 ** n_qubits
    dtype = torch.complex64
    state = torch.zeros(dim, dtype=dtype, device=device)
    state[0] = 1.0 + 0j
    state = state.contiguous()
    for gate_name, q1, q2, angle in rich_tokens:
        U = build_full_unitary_by_name(gate_name, q1, q2, n_qubits, angle, device)
        state = (U @ state.reshape(-1, 1)).reshape(-1).contiguous()
    return state

# ----------------------------
# Differentiable fidelity penalty
# ----------------------------
def fidelity_penalty_differentiable(batch_x: torch.Tensor, batch_y: torch.Tensor, gate_logits: torch.Tensor, q1_logits: torch.Tensor, q2_logits: torch.Tensor, sample_frac: float = SAMPLE_FRACTION_FOR_FID) -> torch.Tensor:
    """
    Compute fidelity penalty for sampled batch elements
    """
    device = gate_logits.device

    # Convert to probabilities
    p_gates = torch.softmax(gate_logits, dim=-1).detach()
    p_q1 = torch.softmax(q1_logits, dim=-1).detach()
    p_q2 = torch.softmax(q2_logits, dim=-1).detach()

    B = batch_x.size(0)
    n_sample = max(1, int(B * sample_frac))
    indices = random.sample(range(B), min(n_sample, B))

    penalties = []
    error_count = 0
    max_errors_to_print = 3

    for i in indices:
        try:
            # Reconstruct window as rich token sequence
            win = batch_x[i].cpu().numpy().tolist()
            seq_rich = []
            for (g_id, q1, q2_m) in win:
                g_name = GATE_VOCAB[int(g_id)]
                q1_i = int(q1)
                q2_i = (int(q2_m) if int(q2_m) != NO_Q_IDX else None)
                angle = (np.pi / 4) if g_name.startswith("rz") else None
                seq_rich.append((g_name, q1_i, q2_i, angle))

            # Simulate state after window
            state_before = simulate_sequence_torch(seq_rich, NUM_QUBITS, device)

            # Ground truth next token
            gy, gq1, gq2_m = batch_y[i].cpu().numpy().tolist()
            gt_gate_name = GATE_VOCAB[int(gy)]
            gt_q1 = int(gq1)
            gt_q2 = (int(gq2_m) if int(gq2_m) != NO_Q_IDX else None)
            gt_angle = (np.pi / 4) if gt_gate_name.startswith("rz") else None

            # Apply ground truth gate
            U_label = build_full_unitary_by_name(gt_gate_name, gt_q1, gt_q2, NUM_QUBITS, gt_angle, device)
            state_label = (U_label @ state_before.reshape(-1, 1)).reshape(-1).contiguous()

            # Build expected state from predicted distribution
            p_g = p_gates[i].clone()
            p_q1_row = p_q1[i].clone()
            p_q2_row = p_q2[i].clone()

            state_pred = torch.zeros_like(state_label, device=device, dtype=torch.complex64)

            G = p_g.size(0)
            Q = p_q1_row.size(0)

            for g_idx in range(G):
                pg = p_g[g_idx]
                if pg.item() < 1e-6:  # Skip very low probability gates
                    continue

                gate_name = GATE_VOCAB[g_idx]

                if gate_name in ["cx", "cz"]:
                    # Two-qubit gate
                    for q1_idx in range(Q):
                        pq1 = p_q1_row[q1_idx]
                        if pq1.item() < 1e-6:
                            continue
                        for q2_idx in range(Q):
                            pq2 = p_q2_row[q2_idx]
                            joint_p = pg * pq1 * pq2
                            if joint_p.item() < 1e-8:
                                continue
                            if q1_idx == q2_idx:  # Invalid: control == target
                                continue
                            U = build_full_unitary_by_name(gate_name, q1_idx, q2_idx, NUM_QUBITS, None, device)
                            contrib = (U @ state_before.reshape(-1, 1)).reshape(-1).contiguous()
                            state_pred = state_pred + joint_p * contrib
                else:
                    # Single-qubit gate
                    for q1_idx in range(Q):
                        pq1 = p_q1_row[q1_idx]
                        pq2_total = p_q2_row.sum()
                        joint_p = pg * pq1 * pq2_total
                        if joint_p.item() < 1e-8:
                            continue
                        U = build_full_unitary_by_name(gate_name, q1_idx, None, NUM_QUBITS, None, device)
                        contrib = (U @ state_before.reshape(-1, 1)).reshape(-1).contiguous()
                        state_pred = state_pred + joint_p * contrib

            # Compute fidelity
            overlap = torch.vdot(state_label, state_pred)
            fidelity = torch.abs(overlap) ** 2
            penalty = 1.0 - fidelity
            penalties.append(penalty)
        except Exception as e:
            error_count += 1
            if error_count <= max_errors_to_print:
                print(f"Warning: Error computing fidelity for sample {i}: {e}")
            elif error_count == max_errors_to_print + 1:
                print(f"... suppressing further fidelity error messages ({error_count} errors so far)")
            continue

    if len(penalties) == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    penalties_tensor = torch.stack(penalties)
    return penalties_tensor.mean()


# ----------------------------
# Structural penalty
# ----------------------------
def structural_penalty(gate_logits: torch.Tensor) -> torch.Tensor:
    """
    Penalty: encourage lower probability mass on non-Clifford gates
    """
    probs = torch.softmax(gate_logits, dim=-1)  # (B, G)
    nonc_mask = torch.zeros(probs.size(1), device=probs.device)
    for idx in NONCLIFFORD_GATES:
        if idx < probs.size(1):
            nonc_mask[idx] = 1.0
    nonc_prob = (probs * nonc_mask[None, :]).sum(dim=-1)  # (B,)
    return nonc_prob.mean()


# ----------------------------
# Eval helper: non-clifford clusters
# ----------------------------
def non_clifford_clusters(seq):
    """Count clusters of consecutive non-Clifford gates"""
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
    """Evaluate non-Clifford clustering and gate accuracy"""
    model.eval()

    ds = dataset_loader.dataset
    N = len(ds)
    if N == 0:
        return

    indices = random.sample(range(N), min(n, N))

    # Metrics
    total_clusters_input = 0
    total_clusters_target = 0
    total_nonclifford_input = 0
    total_nonclifford_target = 0
    correct_gates = 0
    correct_full_tokens = 0

    for idx in indices:
        x, y = ds[idx]

        # Count clusters and non-Cliffords in input window
        input_gates = x[:, 0].tolist()
        clusters_in = non_clifford_clusters(input_gates)
        nonclifford_in = sum(1 for g in input_gates if g in NONCLIFFORD_GATES)

        # Get target token
        target_gate = int(y[0].item())
        target_q1 = int(y[1].item())
        target_q2 = int(y[2].item())

        # Count non-Cliffords in target (just one gate)
        nonclifford_target = 1 if target_gate in NONCLIFFORD_GATES else 0

        # Predict next token
        with torch.no_grad():
            g_logits, q1_logits, q2_logits = model(x.unsqueeze(0).to(DEVICE))
            pred_gate = int(torch.argmax(g_logits, dim=-1).cpu().item())
            pred_q1 = int(torch.argmax(q1_logits, dim=-1).cpu().item())
            pred_q2 = int(torch.argmax(q2_logits, dim=-1).cpu().item())

        # Check accuracy
        if pred_gate == target_gate:
            correct_gates += 1
        if pred_gate == target_gate and pred_q1 == target_q1 and pred_q2 == target_q2:
            correct_full_tokens += 1

        # For cluster evaluation: append predicted gate to sequence
        extended_sequence = input_gates + [pred_gate]
        clusters_out = non_clifford_clusters(extended_sequence)

        total_clusters_input += clusters_in
        total_clusters_target += clusters_out
        total_nonclifford_input += nonclifford_in
        total_nonclifford_target += nonclifford_target

    n_samples = len(indices)
    avg_clusters_in = total_clusters_input / n_samples
    avg_clusters_out = total_clusters_target / n_samples
    avg_nonclifford_in = total_nonclifford_input / (n_samples * WINDOW)
    avg_nonclifford_target = total_nonclifford_target / n_samples
    gate_acc = correct_gates / n_samples * 100
    token_acc = correct_full_tokens / n_samples * 100

    print(f"  Clusters: {avg_clusters_in:.2f} -> {avg_clusters_out:.2f} | "
          f"NonCliff%: window={avg_nonclifford_in*100:.1f}% target={avg_nonclifford_target*100:.1f}% | "
          f"Acc: gate={gate_acc:.1f}% token={token_acc:.1f}%")


# ----------------------------
# Training loop
# ----------------------------
def train():
    print("Starting training on device:", DEVICE)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_fid_pen = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            gate_logits, q1_logits, q2_logits = model(batch_x)

            # Compute supervised CE losses
            gate_target = batch_y[:, 0]
            q1_target = batch_y[:, 1]
            q2_target = batch_y[:, 2]

            loss_gate = ce_loss(gate_logits, gate_target)
            loss_q1 = ce_loss(q1_logits, q1_target)
            loss_q2 = ce_loss(q2_logits, q2_target)
            loss = loss_gate + loss_q1 + loss_q2

            # Add structural penalty
            struct_pen = structural_penalty(gate_logits)
            loss = loss + 0.2 * struct_pen  # Increased weight

            # Differentiable fidelity penalty
            fid_pen = fidelity_penalty_differentiable(batch_x, batch_y, gate_logits, q1_logits, q2_logits, sample_frac=SAMPLE_FRACTION_FOR_FID)
            loss = loss + LAMBDA_FID * fid_pen

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += float(loss.item())
            total_fid_pen += float(fid_pen.item())
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_fid = total_fid_pen / n_batches
        print(f"\nEpoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.4f} avg_fid_pen={avg_fid:.4f}")
        eval_clusters(model, train_loader)

        # Update learning rate based on validation loss
        scheduler.step(avg_loss)
    print("Training finished.")


# ----------------------------
# Test / evaluation
# ----------------------------
def test_model():
    x_list = []
    y_list = []
    for raw, opt in zip(raw_val_sequences, opt_val_sequences):
        length = min(len(raw), len(opt))
        if length <= WINDOW:
            continue
        for i in range(0, length - WINDOW):
            win = raw[i:i + WINDOW]
            nxt = opt[i + WINDOW]

            # Map q2=-1 to NO_Q_IDX for window
            win_mapped = []
            for tok in win:
                g_id, q1, q2 = map(int, tok)
                q2_mapped = q2 if q2 >= 0 else NO_Q_IDX
                win_mapped.append([g_id, q1, q2_mapped])

            # Map q2=-1 to NO_Q_IDX for target
            g_id, q1, q2 = map(int, nxt)
            q2_mapped = q2 if q2 >= 0 else NO_Q_IDX
            x_list.append(win_mapped)
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
    total_correct_full = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            g_logits, q1_logits, q2_logits = model(xb)

            loss_g = ce_loss(g_logits, yb[:, 0])
            loss_q1 = ce_loss(q1_logits, yb[:, 1])
            loss_q2 = ce_loss(q2_logits, yb[:, 2])
            total_loss += float((loss_g + loss_q1 + loss_q2).item()) * xb.size(0)

            _, pred_g = torch.max(g_logits, dim=-1)
            _, pred_q1 = torch.max(q1_logits, dim=-1)
            _, pred_q2 = torch.max(q2_logits, dim=-1)

            total_correct_gates += (pred_g == yb[:, 0]).sum().item()

            # Full token accuracy
            correct_full = ((pred_g == yb[:, 0]) & (pred_q1 == yb[:, 1]) & (pred_q2 == yb[:, 2])).sum().item()
            total_correct_full += correct_full

            total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    gate_accuracy = total_correct_gates / total_samples * 100.0
    full_accuracy = total_correct_full / total_samples * 100.0
    print(f"\nTest Results:")
    print(f"  avg_loss={avg_loss:.4f}")
    print(f"  gate_acc={gate_accuracy:.2f}%")
    print(f"  full_token_acc={full_accuracy:.2f}%")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    train()
    test_model()

    # Save model
    torch.save(model.state_dict(), "quantum_optimizer_model.pth")
    print("Model saved to quantum_optimizer_model.pth")
