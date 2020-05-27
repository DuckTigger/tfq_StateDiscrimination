import cirq
import sympy as sp
from typing import List, Iterable, Union

from subcircuits import SubCircuits


class CircuitLayers:

    @staticmethod
    def create_encoding_circuit(qubits: 'List[cirq.Qid]', symbols: sp.Symbol = None):
        n = len(qubits)
        if symbols is None:
            symbols = sp.symbols('enc0:{}'.format(4 * n))
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(qubits))
        for i, qubit in enumerate(qubits):
            circuit.append(SubCircuits.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3]))
        for i, (qubit_0, qubit_1) in enumerate(zip(qubits, qubits[1:] + [qubits[0]])):
            circuit.append(cirq.CNotPowGate(exponent=symbols[3 * n + i]).on(qubit_0, qubit_1))
        circuit.append(cirq.CNOT(qubits[3], qubits[2]))
        circuit.append(cirq.measure(qubits[2], qubits[3], key='m'))
        return circuit

    @staticmethod
    def create_layers(qubits: List[cirq.Qid], symbols: Iterable[sp.Symbol], level: int,
                      control_circ: bool = False) -> cirq.Circuit:
        layer_qubits = qubits[:-level] if level else qubits
        control_circ = control_circ if level else False
        circuit = cirq.Circuit()
        symbols_0 = sp.symbols(tuple([x.name + '_0' for x in symbols]))
        symbols_1 = sp.symbols(tuple([x.name + '_1' for x in symbols]))
        if control_circ:
            for control, control_sym in enumerate((symbols_0, symbols_1)):
                circuit.append(CircuitLayers.create_control_layer(qubits, control, level, layer_qubits, control_sym))
        else:
            final = level == len(qubits) - 1
            circuit.append(CircuitLayers.create_non_control_layer(final, layer_qubits, symbols))
        circuit.append(cirq.measure(layer_qubits[-1], key='m{}'.format(level)))
        return circuit

    @staticmethod
    def create_non_control_layer(final: bool, layer_qubits: List[cirq.Qid],
                                 symbols: Union[Iterable[sp.Symbol], sp.Symbol]) -> cirq.Circuit:
        circuit = cirq.Circuit()
        n = len(layer_qubits)
        for i, qubit in enumerate(layer_qubits):
            circuit.append(SubCircuits.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3]))
        if not final:
            for i, (qubit_0, qubit_1) in enumerate(zip(layer_qubits, layer_qubits[1:] + [layer_qubits[0]])):
                circuit.append(
                    cirq.CNotPowGate(exponent=symbols[3 * n + i]).on(qubit_0, qubit_1))
        return circuit

    @staticmethod
    def create_control_layer(qubits: 'List[cirq.Qid]', control: int, level: int,
                             layer_qubits: List[cirq.Qid], symbols: sp.Symbol) -> cirq.Circuit:
        circuit = cirq.Circuit()
        control_qubit = qubits[-level]
        n = len(layer_qubits)
        for i, qubit in enumerate(layer_qubits):
            circuit.append(SubCircuits.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3], control_qubit, control))
        if not level == len(qubits) - 1:
            for i, (qubit_0, qubit_1) in enumerate(zip(layer_qubits, layer_qubits[1:] + [layer_qubits[0]])):
                circuit.append(
                    cirq.CNotPowGate(exponent=symbols[3 * n + i]).on(qubit_0, qubit_1).controlled_by(
                        control_qubit, control_values=[control]))
        return circuit

    @staticmethod
    def leakage_qutrit_layers(work_qubits: 'List[cirq.Qid]', ancilla_qubits: 'List[cirq.Qid]',
                            readout_qubits: 'List[cirq.Qid]',
                            level: int, symbols: List[sp.Symbol],
                              train_qutrits: bool, constant_leakage: float = None) -> 'cirq.Circuit':
        final = level == (len(work_qubits) - 1)
        (work_qubits, ancilla_qubits, readout_qubits) = \
            (work_qubits[:-level], ancilla_qubits[:-level], readout_qubits[:-level]) if level \
                else (work_qubits, ancilla_qubits, readout_qubits)
        symbols_leakage = sp.symbols(tuple([x.name + '_leak' for x in symbols])) if constant_leakage is None \
            else [constant_leakage for _ in range(len(work_qubits))]

        if train_qutrits:
            yield CircuitLayers.create_qutrit_layer(work_qubits, ancilla_qubits, final, symbols)
        else:
            yield CircuitLayers.create_non_control_layer(final, work_qubits, symbols)

        for work, ancilla, symbol in zip(work_qubits, ancilla_qubits, symbols_leakage):
            yield SubCircuits.leakage(work, ancilla, symbol)
        yield SubCircuits.quantum_OR(work_qubits[-1], ancilla_qubits[-1], readout_qubits[-1])
        yield cirq.measure(readout_qubits[-1], key='m{}'.format(level))

    @staticmethod
    def create_qutrit_layer(layer_qubits: 'List[cirq.Qid]', layer_ancilla: 'List[cirq.Qid]',
                            final: bool, symbols: List[sp.Symbol]) -> 'cirq.Circuit':
        n = len(layer_qubits)
        for i, (qubit, ancilla) in enumerate(zip(layer_qubits, layer_ancilla)):
            yield SubCircuits.qutrit_unitary(qubit, ancilla, symbols[8 * i: 8 * i + 8])
        if not final:
            for i, (control_qubit, control_ancilla,
                    target_qubit, target_ancilla) in enumerate(zip(layer_qubits, layer_ancilla,
                                                                   layer_qubits[1:] + [layer_qubits[0]],
                                                                   layer_ancilla[1:] + [layer_ancilla[0]])):
                yield SubCircuits.qutrit_CSUM(control_ancilla, control_qubit, target_ancilla, target_qubit,
                                                symbols[8 * n + i])
