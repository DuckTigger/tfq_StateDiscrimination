import sympy as sp
import tensorflow_quantum as tfq
import tensorflow as tf
from math import ceil
import cirq

from circuit_layers import CircuitLayers


class QutritModel:

    def __init__(self, n_work_qubits: int, constant_leakage: float = None):
        self.qubits = cirq.GridQubit.rect(ceil(3 * n_work_qubits / 2), 2)
        self.work_qubits = self.qubits[:n_work_qubits]
        self.ancilla_qubits = self.qubits[n_work_qubits:2 * n_work_qubits]
        self.readout_qubits = self.qubits[2 * n_work_qubits: 3 * n_work_qubits]
        self.n_layers = n_work_qubits
        self.constant_leakage = constant_leakage

    def qutrit_discrimination_circuit(self):
        output = cirq.Circuit()
        for level in range(self.n_layers):
            symbols = sp.symbols('layer{}_0:{}'.format(level, 9 * self.n_layers - level))
            output += CircuitLayers.leakage_qutrit_layers(self.work_qubits, self.ancilla_qubits, self.readout_qubits,
                                                          level, symbols, True, self.constant_leakage)
        return output

    def qutrit_model(self, backend: 'cirq.Simulator' = None):
        circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        qutrit_circuit = self.qutrit_discrimination_circuit()
        readout_ops = [cirq.Z(q) for q in self.readout_qubits]
        qutrit_pqc = tfq.layers.PQC(qutrit_circuit, readout_ops, backend=backend)(circuit_input)
        return tf.keras.Model(inputs=[circuit_input], outputs=[qutrit_pqc])
