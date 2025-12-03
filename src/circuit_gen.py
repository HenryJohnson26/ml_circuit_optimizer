# src/circuit_gen.py
import random
from typing import List, Tuple
from qiskit import QuantumCircuit

# small vocabulary choices
CLIFFORD_SINGLE = ["h", "s", "sdg", "x", "y", "z"]
CLIFFORD_TWOQ = ["cx", "cz"]
NON_CLIFFORD_SINGLE = ["t", "tdg", "rz_pi_4"]  # rz_pi_4 denotes RZ(pi/4)

def _maybe_add_parameterized_gate(qc: QuantumCircuit, gate: str, q: int):
    if gate == "rz_pi_4":
        qc.rz(3.141592653589793 / 4.0, q)
    elif gate == "sdg":
        qc.sdg(q)
    elif gate == "tdg":
        qc.tdg(q)
    else:
        # other single qubit gates that qiskit supports by method name
        getattr(qc, gate)(q)

def random_circuit(n_qubits: int,
                   depth: int,
                   non_clifford_density: float = 0.2,
                   two_qubit_prob: float = 0.35,
                   seed: int = None) -> QuantumCircuit:
    """
    Generate a random circuit.
    - non_clifford_density: fraction of single-qubit gates chosen from NON_CLIFFORD_SINGLE
    - two_qubit_prob: probability that the layer uses a two-qubit gate (otherwise single-qubit gate)
    """
    if seed is not None:
        random.seed(seed)
    qc = QuantumCircuit(n_qubits)
    for layer in range(depth):
        # randomly choose positions for gates in this layer
        occupied = set()
        for q in range(n_qubits):
            if q in occupied:
                continue
            if random.random() < two_qubit_prob and q < n_qubits - 1 and (q + 1) not in occupied:
                gate = random.choice(CLIFFORD_TWOQ + ["cx"])  # include cx/cz choices (cx common)
                if gate == "cx":
                    qc.cx(q, q + 1)
                elif gate == "cz":
                    qc.cz(q, q + 1)
                occupied.add(q)
                occupied.add(q + 1)
            else:
                # single qubit gate
                if random.random() < non_clifford_density:
                    gate = random.choice(NON_CLIFFORD_SINGLE)
                else:
                    gate = random.choice(CLIFFORD_SINGLE)
                _maybe_add_parameterized_gate(qc, gate, q)
                occupied.add(q)
    return qc
