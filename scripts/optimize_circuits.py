# scripts/optimize_circuits.py
import glob
import os
from qiskit import QuantumCircuit
from qiskit import qasm3
from qiskit.qasm3 import loads as qasm3_loads # <-- IMPORT THE QASM3 LOADER
from src.qiskit_passes import run_custom_passes, transpile_optimize_simple

RAW_DIR = "data/raw"
OPT_DIR = "data/optimized"
os.makedirs(OPT_DIR, exist_ok=True)

def load_qasm(path):
    """Loads a QuantumCircuit from an OpenQASM 3.0 file."""
    # FIX: Use the QASM 3.0 loader (qasm3_loads) instead of the QASM 2.0 default 
    # used by QuantumCircuit.from_qasm_str()
    return qasm3_loads(open(path).read())

def optimize_all(use_custom_passes=True):
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.qasm")))
    for path in files:
        qc = load_qasm(path)
        if use_custom_passes:
            opt = run_custom_passes(qc)
        else:
            opt = transpile_optimize_simple(qc, optimization_level=1)
        # save
        base = os.path.basename(path)
        outpath = os.path.join(OPT_DIR, base.replace(".qasm", "_opt.qasm"))
        with open(outpath, "w") as f:
            f.write(qasm3.dumps(opt))
    print(f"Optimized {len(files)} circuits -> {OPT_DIR}")

if __name__ == "__main__":
    optimize_all(use_custom_passes=True)
