import sympy as sp
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
from math import ceil
from typing import List
from circuit_layers import CircuitLayers


class LeakageModels:

    def __init__(self, n_work_qubits: int, n_data_qubits: int = 2, train_on_data: bool = False,
                 constant_leakage: float = None):
        if train_on_data:
            n_data_qubits = 0
        self.qubits = cirq.GridQubit.rect(ceil(((3 * n_work_qubits) + n_data_qubits) / 2), 2)
        self.data_qubits = self.qubits[:n_data_qubits]
        self.work_qubits = self.qubits[n_data_qubits:n_data_qubits + n_work_qubits]
        self.ancilla_qubits = self.qubits[n_data_qubits + n_work_qubits: n_data_qubits + 2 * n_work_qubits]
        self.readout_qubits = self.qubits[n_data_qubits + 2 * n_work_qubits: n_data_qubits + 3 * n_work_qubits]
        self.n_layers = n_work_qubits
        self.constant_leakage = constant_leakage
        self.train_on_data = train_on_data

    def leaky_discrimination_circuit(self):
        output = cirq.Circuit()
        for level in range(self.n_layers):
            symbols = sp.symbols('layer{}_0:{}'.format(level, 4 * self.n_layers - level))
            output += CircuitLayers.leakage_qutrit_layers(self.work_qubits, self.ancilla_qubits, self.readout_qubits,
                                                          level, symbols, False, self.constant_leakage)
        return output

    def leaky_model(self, backend: 'cirq.Simulator' = None, repetitions: int = None):
        circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        leaky_circuit = self.leaky_discrimination_circuit()
        readout_ops = [cirq.Z(q) for q in self.readout_qubits]
        leaky_pqc = tfq.layers.PQC(leaky_circuit, readout_ops, backend=backend, repetitions=repetitions)(circuit_input)
        return tf.keras.Model(inputs=[circuit_input], outputs=[leaky_pqc])
