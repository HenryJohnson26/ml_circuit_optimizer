import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from qiskit.qasm3 import dumps as qasm3_dumps

def transpile_optimize_simple(qc: QuantumCircuit, basis_gates=None, optimization_level: int = 1) -> QuantumCircuit:
    """
    Use qiskit.compiler.transpile with a specified optimization level (Qiskit 1.0+).
    
    This function uses the standard, fast transpilation path, automatically 
    including gate cancellation and 1-qubit gate merging at optimization_level >= 1.
    
    Args:
        qc (QuantumCircuit): The input quantum circuit.
        basis_gates (list[str], optional): Target basis gates (e.g., ['rz', 'sx', 'x', 'cx']).
        optimization_level (int): Optimization level (0-3). Defaults to 1.
        
    Returns:
        QuantumCircuit: The transpiled and optimized circuit.
    """
    # Note: `qiskit.compiler.transpile` is the standard function in Qiskit 1.0+
    return transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level)

def run_custom_passes(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Build a custom PassManager with specific passes for local circuit simplification 
    (Qiskit 1.0+).
    
    This custom set focuses on local gate simplification, which is crucial 
    for generating consistent tokens (e.g., for ML tokenization tasks).
    
    Args:
        qc (QuantumCircuit): The input quantum circuit.
        
    Returns:
        QuantumCircuit: The circuit after applying custom optimization passes.
    """
    pm = PassManager()
    
    # 1. Merge single qubit gates (e.g., Rz(a) Rz(b) -> Rz(a+b))
    pm.append(Optimize1qGates())
    
    # 2. Cancel commuting 2q gates where possible (e.g., handles XX-YY cancellation 
    #    and implicitly handles CX CX -> I, or H CX H -> CZ simplification).
    pm.append(CommutativeCancellation())
    
    optimized = pm.run(qc)
    return optimized

