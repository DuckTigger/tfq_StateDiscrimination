import cirq
import numpy as np
import sympy as sy
import tensorflow_quantum as tfq
import tensorflow as tf


class EncodeState:

    def __init__(self, n_qubits: int):
        self.qubits = cirq.GridQubit.rect(int(n_qubits/2), 2)
        self.n = n_qubits

    def create_encoding_layers(self, symbols: sy.Symbol = None):
        if symbols is None:
            symbols = sy.symbols('enc0:{}'.format(4 * self.n))
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(self.qubits))
        for i, qubit in enumerate(self.qubits):
            circuit.append(self.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3]))
        for i, (qubit_0, qubit_1) in enumerate(zip(self.qubits, self.qubits[1:] + [self.qubits[0]])):
            circuit.append(cirq.CX(qubit_0, qubit_1)**symbols[3 * self.n + i])
        return circuit

    @staticmethod
    def one_qubit_unitary(qubit: cirq.Qid, symbols: sy.Symbol):
        return cirq.Circuit([cirq.X(qubit)**symbols[0],
                             cirq.Y(qubit)**symbols[1],
                             cirq.Z(qubit)**symbols[2]])

    def ent_ops(self):
        return cirq.Circuit(cirq.CNOT(q1, q2) for q1, q2 in zip(self.qubits, self.qubits[1:] + [self.qubits[0]]))

    def encode_state(self):
        symbols = sy.symbols('enc0:{}'.format(4 * self.n))
        encoding_circuit = self.create_encoding_layers(symbols)
        encoding_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        encoding_layer = tfq.layers.AddCircuit()(encoding_input, prepend=self.ent_ops())
        readout_ops = cirq.PauliString(1, cirq.Z(self.qubits[2]), cirq.Z(self.qubits[3]))
        encoding_model = tfq.layers.PQC(encoding_circuit, readout_ops)(encoding_layer)
        return tf.keras.Model(inputs=[encoding_input], outputs=[encoding_model])

    def decode_state(self):
        pass
