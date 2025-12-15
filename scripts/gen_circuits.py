import json
from qiskit import QuantumCircuit
from qiskit import qasm3
from src.circuit_gen import random_circuit
import os

RAW_DIR = "data/raw"

def save_qasm(qc: QuantumCircuit, fname: str):
    with open(fname, "w") as f:
        f.write(qasm3.dumps(qc))

def generate_and_save(n_samples=100, n_qubits=4, depth=40, non_clifford_density=0.2, seed0=0):
    os.makedirs(RAW_DIR, exist_ok=True)
    for i in range(n_samples):
        seed = seed0 + i
        qc = random_circuit(n_qubits=n_qubits, depth=depth, non_clifford_density=non_clifford_density, seed=seed)
        fname = os.path.join(RAW_DIR, f"circuit_{n_qubits}q_d{depth}_id{i}.qasm")
        save_qasm(qc, fname)
    print(f"Saved {n_samples} circuits to {RAW_DIR}")

if __name__ == "__main__":
    generate_and_save(n_samples=1000, n_qubits=4, depth=50, non_clifford_density=0.25)
