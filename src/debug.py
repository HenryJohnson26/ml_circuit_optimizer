#!/usr/bin/env python3

import json
import torch
from pathlib import Path

# Load data
DATA_DIR = "data/tokenized"
WINDOW = 32

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
print(f"Loaded {len(raw_sequences)} circuits")

# Check for maximum values in the data
max_gate_id = -1
max_q1 = -1
max_q2 = -1
min_q2 = float('inf')

for raw, opt in zip(raw_sequences, opt_sequences):
    for seq in [raw, opt]:
        for token in seq:
            gate_id, q1, q2 = token
            max_gate_id = max(max_gate_id, gate_id)
            max_q1 = max(max_q1, q1)
            if q2 >= 0:
                max_q2 = max(max_q2, q2)
            min_q2 = min(min_q2, q2)

print(f"\nData Statistics:")
print(f"  max_gate_id: {max_gate_id}")
print(f"  max_q1: {max_q1}")
print(f"  max_q2: {max_q2}")
print(f"  min_q2: {min_q2}")

# Infer parameters
NUM_QUBITS = max(max_q1, max_q2) + 1
NO_Q_IDX = NUM_QUBITS
VOCAB_SIZE = max_gate_id + 1

print(f"\nInferred Parameters:")
print(f"  NUM_QUBITS: {NUM_QUBITS}")
print(f"  NO_Q_IDX: {NO_Q_IDX}")
print(f"  VOCAB_SIZE: {VOCAB_SIZE}")

# Build dataset and check for issues
def build_dataset_lists(raw_seqs, opt_seqs, window=WINDOW):
    X_list = []
    Y_list = []
    for raw, opt in zip(raw_seqs, opt_seqs):
        length = min(len(raw), len(opt))
        if length <= window:
            continue
        for i in range(0, length - window):
            win = raw[i:i + window]
            nxt = opt[i + window]
            g_id, q1, q2 = map(int, nxt)
            q2_mapped = q2 if q2 >= 0 else NO_Q_IDX
            X_list.append(win)
            Y_list.append([g_id, q1, q2_mapped])
    return X_list, Y_list

X_train_list, Y_train_list = build_dataset_lists(raw_sequences[:1], opt_sequences[:1], WINDOW)

print(f"\nDataset samples: {len(X_train_list)}")

if len(X_train_list) > 0:
    # Check first sample
    print(f"\nFirst X sample (first 3 tokens):")
    for j in range(min(3, len(X_train_list[0]))):
        print(f"  Token {j}: {X_train_list[0][j]}")

    print(f"\nFirst Y sample: {Y_train_list[0]}")

    # Convert to tensor and check ranges
    X_train = torch.tensor(X_train_list, dtype=torch.long)
    Y_train = torch.tensor(Y_train_list, dtype=torch.long)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    print(f"\nX_train statistics:")
    print(f"  gate_ids - min: {X_train[:, :, 0].min()}, max: {X_train[:, :, 0].max()}")
    print(f"  q1 - min: {X_train[:, :, 1].min()}, max: {X_train[:, :, 1].max()}")
    print(f"  q2 - min: {X_train[:, :, 2].min()}, max: {X_train[:, :, 2].max()}")

    print(f"\nY_train statistics:")
    print(f"  gate_id - min: {Y_train[:, 0].min()}, max: {Y_train[:, 0].max()}")
    print(f"  q1 - min: {Y_train[:, 1].min()}, max: {Y_train[:, 1].max()}")
    print(f"  q2 - min: {Y_train[:, 2].min()}, max: {Y_train[:, 2].max()}")

    # Check for out-of-bounds values
    print(f"\n=== CRITICAL CHECKS ===")

    # Gate IDs should be < VOCAB_SIZE
    bad_gates_x = (X_train[:, :, 0] >= VOCAB_SIZE).any()
    bad_gates_y = (Y_train[:, 0] >= VOCAB_SIZE).any()
    print(f"Gate IDs >= VOCAB_SIZE ({VOCAB_SIZE}): X={bad_gates_x}, Y={bad_gates_y}")

    # Q1 should be < NUM_QUBITS
    bad_q1_x = (X_train[:, :, 1] >= NUM_QUBITS).any()
    bad_q1_y = (Y_train[:, 1] >= NUM_QUBITS).any()
    print(f"Q1 >= NUM_QUBITS ({NUM_QUBITS}): X={bad_q1_x}, Y={bad_q1_y}")

    # Q2 should be < NUM_QUBITS + 1 (includes NO_Q_IDX)
    bad_q2_x = (X_train[:, :, 2] >= NUM_QUBITS + 1).any()
    bad_q2_y = (Y_train[:, 2] >= NUM_QUBITS + 1).any()
    print(f"Q2 >= NUM_QUBITS+1 ({NUM_QUBITS + 1}): X={bad_q2_x}, Y={bad_q2_y}")

    # Check for negative values (except -1 which should be mapped)
    neg_q1_x = (X_train[:, :, 1] < 0).any()
    neg_q1_y = (Y_train[:, 1] < 0).any()
    neg_q2_x = (X_train[:, :, 2] < 0).any()
    neg_q2_y = (Y_train[:, 2] < 0).any()
    print(f"Negative Q1: X={neg_q1_x}, Y={neg_q1_y}")
    print(f"Negative Q2 (should be False): X={neg_q2_x}, Y={neg_q2_y}")

    if bad_gates_x or bad_gates_y:
        print("\n!!! FOUND BAD GATE IDS !!!")
        if bad_gates_y:
            bad_idx = (Y_train[:, 0] >= VOCAB_SIZE).nonzero()
            print(f"Bad Y gate indices: {bad_idx[:5]}")
            print(f"Bad Y gate values: {Y_train[bad_idx[:5], 0]}")

    if bad_q1_x or bad_q1_y:
        print("\n!!! FOUND BAD Q1 VALUES !!!")
        if bad_q1_y:
            bad_idx = (Y_train[:, 1] >= NUM_QUBITS).nonzero()
            print(f"Bad Y q1 indices: {bad_idx[:5]}")
            print(f"Bad Y q1 values: {Y_train[bad_idx[:5], 1]}")

    if bad_q2_x or bad_q2_y:
        print("\n!!! FOUND BAD Q2 VALUES !!!")
        if bad_q2_y:
            bad_idx = (Y_train[:, 2] >= NUM_QUBITS + 1).nonzero()
            print(f"Bad Y q2 indices: {bad_idx[:5]}")
            print(f"Bad Y q2 values: {Y_train[bad_idx[:5], 2]}")

print("\n=== Checking opt_tokens for 'rz' gates ===")
# Check if there are 'rz' gates (not 'rz_pi_4') in opt_tokens
for i, f in enumerate(json_files[:3]):
    with open(f, "r") as fp:
        d = json.load(fp)
        opt_tokens = d.get("opt_tokens", [])
        for j, token in enumerate(opt_tokens):
            if isinstance(token, list) and len(token) > 0:
                gate_name = token[0]
                if gate_name == "rz":
                    print(f"File {i}: Found 'rz' gate at index {j}: {token}")
                    print(f"  Corresponding opt_int: {d['opt_ints'][j]}")
