# scripts/tokenize.py
import glob
import json
import os
from src.tokenizers import circuit_to_token_seq, token_seq_to_ints
from qiskit import QuantumCircuit
from qiskit.qasm3 import loads as qasm3_loads # <-- IMPORT THE QASM3 LOADER

RAW_DIR = "data/raw"
OPT_DIR = "data/optimized"
OUT_DIR = "data/tokenized"
os.makedirs(OUT_DIR, exist_ok=True)

def load_qasm(path):
    """Loads a QuantumCircuit from an OpenQASM 3.0 file."""
    # FIX: Use the QASM 3.0 loader (qasm3_loads) instead of the QASM 2.0 default 
    # used by QuantumCircuit.from_qasm_str()
    return qasm3_loads(open(path).read())

def tokenize_pair(raw_qasm_path, opt_qasm_path, outpath_prefix):
    raw_qc = load_qasm(raw_qasm_path)
    opt_qc = load_qasm(opt_qasm_path)
    raw_tokens = circuit_to_token_seq(raw_qc)
    opt_tokens = circuit_to_token_seq(opt_qc)
    raw_ints = token_seq_to_ints(raw_tokens, n_qubits=raw_qc.num_qubits)
    opt_ints = token_seq_to_ints(opt_tokens, n_qubits=raw_qc.num_qubits)
    payload = {
        "raw_tokens": raw_tokens,
        "opt_tokens": opt_tokens,
        "raw_ints": raw_ints,
        "opt_ints": opt_ints,
        "n_qubits": raw_qc.num_qubits
    }
    with open(outpath_prefix, "w") as f:
        json.dump(payload, f, default=str)
    return payload

def tokenize_all():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.qasm")))
    for raw_path in raw_files:
        base = os.path.basename(raw_path).replace(".qasm", "")
        opt_path = os.path.join(OPT_DIR, base + "_opt.qasm")
        if not os.path.exists(opt_path):
            print("Warning: missing optimized file for", raw_path)
            continue
        outp = os.path.join(OUT_DIR, base + ".json")
        tokenize_pair(raw_path, opt_path, outp)
    print("Tokenized dataset ->", OUT_DIR)

if __name__ == "__main__":
    tokenize_all()
