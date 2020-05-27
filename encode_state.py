import cirq
import numpy as np
import sympy as sp
import tensorflow_quantum as tfq
import tensorflow as tf
from typing import Iterable, List, Union

from circuit_layers import CircuitLayers


class EncodeState:

    def __init__(self, n_qubits: int, n_data_qubits: int = 2):
        self.qubits = cirq.GridQubit.rect(int(n_qubits/2), 2)
        self.n = n_qubits
        self.n_data = n_data_qubits

    def discrimination_circuit(self, control_on_measurement: bool = False):
        output = cirq.Circuit()
        for i in range(self.n - self.n_data):
            symbols = sp.symbols('layer{}_0:{}'.format(i, 4 * self.n - i))
            output.append(CircuitLayers.create_layers(self.qubits, symbols, i, control_on_measurement))
        return output

    def encode_state_PQC(self):
        symbols = sp.symbols('enc0:{}'.format(4 * self.n))
        encoding_circuit = CircuitLayers.create_encoding_circuit(self.qubits, symbols)
        encoding_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        encoding_layer = tfq.layers.AddCircuit()(encoding_input, prepend=CircuitLayers.ent_ops(self.qubits))
        readout_ops = [cirq.Z(self.qubits[2]), cirq.Z(self.qubits[3])]
        encoding_model = tfq.layers.PQC(encoding_circuit, readout_ops)(encoding_layer)
        return tf.keras.Model(inputs=[encoding_input], outputs=[encoding_model])

    def discrimination_model(self, control: bool = False, backend: 'cirq.Simulator' = None):
        discrimination_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        discrimination_circuit = self.discrimination_circuit(control)
        measurement_qubits = self.qubits[::-1][:-self.n_data] if self.n_data else self.qubits
        readout_ops = [cirq.Z(qubit) for qubit in measurement_qubits]
        discrimination_pqc = tfq.layers.PQC(discrimination_circuit, readout_ops, backend=backend)(discrimination_input)
        return tf.keras.Model(inputs=[discrimination_input], outputs=[discrimination_pqc])
