import sympy as sp
import tensorflow_quantum as tfq
import cirq


class LeakageCircuit:

    @staticmethod
    def leakage(leaky_qubit : 'cirq.Qid', ancilla_qubit: 'cirq.Qid', symbol: 'sp.Symbol'):
        circuit = cirq.Circuit()
        circuit.append(cirq.CNOT(control=ancilla_qubit, target=leaky_qubit))
        circuit.append((cirq.Y ** symbol)(ancilla_qubit))
        circuit.append(cirq.CNOT(control=leaky_qubit, target=ancilla_qubit))
        circuit.append(((cirq.Y ** symbol) ** -1)(ancilla_qubit))
        circuit.append(cirq.CNOT(control=leaky_qubit, target=ancilla_qubit))
        circuit.append(cirq.CNOT(control=ancilla_qubit, target=leaky_qubit))
        return circuit