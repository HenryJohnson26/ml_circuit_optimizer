# src/tokenizers.py
from typing import List, Tuple, Dict, Any
from qiskit import QuantumCircuit
import math

# define gate vocabulary (include both clifford and non)
GATE_VOCAB = [
    "h", "s", "sdg", "x", "y", "z", "t", "tdg", "rz_pi_4",
    "cx", "cz"
]
GATE_TO_ID = {g: i for i, g in enumerate(GATE_VOCAB)}
ID_TO_GATE = {i: g for g, i in GATE_TO_ID.items()}

def circuit_to_token_seq(qc: QuantumCircuit) -> List[Tuple[str, Tuple[int, ...], Any]]:
    """
    Convert a QuantumCircuit into a list of tokens:
    (gate_name, (qubit indices...), parameter)
    parameter is None for most gates; for rz it will be the rotation angle (float)
    """
    tokens = []
    for inst, qargs, cargs in qc.data:
        name = inst.name
        qinds = tuple([q._index for q in qargs])
        param = None
        if name == "rz":
            # try to canonicalize to rz_pi_4 if close to pi/4 multiples
            val = float(inst.params[0])
            # normalize to representative strings
            if abs(val - math.pi/4) < 1e-9:
                name = "rz_pi_4"
                param = val
            else:
                param = val
        tokens.append((name, qinds, param))
    return tokens

def token_seq_to_ints(token_seq, n_qubits, max_window_size=1):
    """
    Convert token sequence to integer features:
    For simplicity, encode each token as: [gate_id, qubit0, qubit1 (or -1)]
    Returns a list of integer tuples.
    """
    ints = []
    for (name, qinds, param) in token_seq:
        gate_id = GATE_TO_ID.get(name, None)
        if gate_id is None:
            # unknown gate: map to a special token (or skip)
            continue
        # pad qubit indices to length 2 for consistency
        if len(qinds) == 1:
            q0 = qinds[0]
            q1 = -1
        elif len(qinds) == 2:
            q0, q1 = qinds
        else:
            # skip weird multi-qubit ops
            continue
        ints.append((gate_id, q0, q1))
    return ints
