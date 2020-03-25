import cirq
import numpy as np
import sympy as sp
import tensorflow_quantum as tfq
import tensorflow as tf
from typing import Iterable, List, Union


class EncodeState:

    def __init__(self, n_qubits: int, n_data_qubits: int = 2):
        self.qubits = cirq.GridQubit.rect(int(n_qubits/2), 2)
        self.n = n_qubits
        self.n_data = n_data_qubits

    def create_encoding_circuit(self, symbols: sp.Symbol = None):
        if symbols is None:
            symbols = sp.symbols('enc0:{}'.format(4 * self.n))
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(self.qubits))
        for i, qubit in enumerate(self.qubits):
            circuit.append(self.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3]))
        for i, (qubit_0, qubit_1) in enumerate(zip(self.qubits, self.qubits[1:] + [self.qubits[0]])):
            circuit.append(cirq.CNotPowGate(exponent=symbols[3 * self.n + i]).on(qubit_0, qubit_1))
        circuit.append(cirq.CNOT(self.qubits[3], self.qubits[2]))
        circuit.append(cirq.measure(self.qubits[2], self.qubits[3], key='m'))
        return circuit

    def create_layers(self, symbols: Iterable[sp.Symbol], level: int, control_circ: bool = False) -> cirq.Circuit:
        layer_qubits = self.qubits[:-level] if level else self.qubits
        control_circ = control_circ if level else False
        circuit = cirq.Circuit()
        symbols_0 = sp.symbols(tuple([x.name + '_0' for x in symbols]))
        symbols_1 = sp.symbols(tuple([x.name + '_1' for x in symbols]))
        if control_circ:
            for control, control_sym in enumerate((symbols_0, symbols_1)):
                circuit.append(self.create_control_layer(control, level, layer_qubits, control_sym))
        else:
            circuit.append(self.create_non_control_layer(level, layer_qubits, symbols))
        circuit.append(cirq.measure(layer_qubits[-1], key='m{}'.format(level)))
        return circuit

    def create_control_layer(self, control: int, level: int,
                             layer_qubits: List[cirq.Qid], symbols: sp.Symbol) -> cirq.Circuit:
        circuit = cirq.Circuit()
        control_qubit = self.qubits[-level]
        n = len(layer_qubits)
        for i, qubit in enumerate(layer_qubits):
            circuit.append(self.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3], control_qubit, control))
        if not level == len(self.qubits) - 1:
            for i, (qubit_0, qubit_1) in enumerate(zip(layer_qubits, layer_qubits[1:] + [layer_qubits[0]])):
                circuit.append(
                    cirq.CNotPowGate(exponent=symbols[3 * n + i]).on(qubit_0, qubit_1).controlled_by(
                        control_qubit, control_values=[control]))
        return circuit

    def create_non_control_layer(self, level: int, layer_qubits: List[cirq.Qid],
                                 symbols: Union[Iterable[sp.Symbol], sp.Symbol]) -> cirq.Circuit:
        circuit = cirq.Circuit()
        n = len(layer_qubits)
        for i, qubit in enumerate(layer_qubits):
            circuit.append(self.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3]))
        if not level == len(self.qubits) - 1:
            for i, (qubit_0, qubit_1) in enumerate(zip(layer_qubits, layer_qubits[1:] + [layer_qubits[0]])):
                circuit.append(
                    cirq.CNotPowGate(exponent=symbols[3 * n + i]).on(qubit_0, qubit_1))
        return circuit

    @staticmethod
    def one_qubit_unitary(qubit: cirq.Qid, symbols: sp.Symbol, control_qubit: cirq.Qid = None, control: int = None) -> cirq.Circuit:
        if control_qubit is None:
            return cirq.Circuit([cirq.X(qubit)**symbols[0],
                                 cirq.Y(qubit)**symbols[1],
                                 cirq.Z(qubit)**symbols[2]])
        else:
            return cirq.Circuit([cirq.XPowGate(exponent=symbols[0]).on(qubit).controlled_by(
                                     control_qubit, control_values=[control]),
                                 cirq.YPowGate(exponent=symbols[1]).on(qubit).controlled_by(
                                     control_qubit, control_values=[control]),
                                 cirq.ZPowGate(exponent=symbols[2]).on(qubit).controlled_by(
                                     control_qubit, control_values=[control])])

    def ent_ops(self):
        return cirq.Circuit(cirq.CNOT(q1, q2) for q1, q2 in zip(self.qubits, self.qubits[1:] + [self.qubits[0]]))

    def discrimination_circuit(self, control_on_measurement: bool = False):
        output = cirq.Circuit()
        for i in range(self.n - self.n_data):
            symbols = sp.symbols('layer{}_0:{}'.format(i, 4 * self.n - i))
            output.append(self.create_layers(symbols, i, control_on_measurement))
        return output

    def encode_state_PQC(self):
        symbols = sp.symbols('enc0:{}'.format(4 * self.n))
        encoding_circuit = self.create_encoding_circuit(symbols)
        encoding_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        encoding_layer = tfq.layers.AddCircuit()(encoding_input, prepend=self.ent_ops())
        readout_ops = [cirq.Z(self.qubits[2]), cirq.Z(self.qubits[3])]
        encoding_model = tfq.layers.PQC(encoding_circuit, readout_ops)(encoding_layer)
        return tf.keras.Model(inputs=[encoding_input], outputs=[encoding_model])

    def discrimination_model(self, control: bool = False):
        discrimination_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        discrimination_circuit = self.discrimination_circuit(control)
        measurement_qubits = self.qubits[::-1][:-self.n_data] if self.n_data else self.qubits
        readout_ops = [cirq.Z(qubit) for qubit in measurement_qubits]
        discrimination_pqc = tfq.layers.PQC(discrimination_circuit, readout_ops)(discrimination_input)
        return tf.keras.Model(inputs=[discrimination_input], outputs=[discrimination_pqc])
