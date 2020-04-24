import cirq
import sympy as sp
from typing import List, Iterable, Union


class ModelCircuits:

    @staticmethod
    def create_encoding_circuit(qubits: 'List[cirq.Qid]', symbols: sp.Symbol = None):
        n = len(qubits)
        if symbols is None:
            symbols = sp.symbols('enc0:{}'.format(4 * n))
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(qubits))
        for i, qubit in enumerate(qubits):
            circuit.append(ModelCircuits.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3]))
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
                circuit.append(ModelCircuits.create_control_layer(qubits, control, level, layer_qubits, control_sym))
        else:
            final = level == len(qubits) - 1
            circuit.append(ModelCircuits.create_non_control_layer(final, layer_qubits, symbols))
        circuit.append(cirq.measure(layer_qubits[-1], key='m{}'.format(level)))
        return circuit

    @staticmethod
    def create_leakage_layers(work_qubits: 'List[cirq.Qid]', ancilla_qubits: 'List[cirq.Qid]',
                              readout_qubits: 'List[cirq.Qid]', symbols: 'Iterable[sp.Symbol]', level: int,
                              constant_leakage: float = None):
        final = level == (len(work_qubits) - 1)
        (work_qubits, ancilla_qubits, readout_qubits) = \
            (work_qubits[:-level], ancilla_qubits[:-level], readout_qubits[:-level]) if level \
                else (work_qubits, ancilla_qubits, readout_qubits)

        symbols_leakage = sp.symbols(tuple([x.name + '_leak' for x in symbols])) if constant_leakage is None \
            else [constant_leakage for _ in range(len(work_qubits))]

        yield ModelCircuits.create_non_control_layer(final, work_qubits, symbols)
        for work, ancilla, symbol in zip(work_qubits, ancilla_qubits, symbols_leakage):
            yield ModelCircuits.leakage(work, ancilla, symbol)
        yield ModelCircuits.quantum_OR(work_qubits[-1], ancilla_qubits[-1], readout_qubits[-1])
        yield cirq.measure(readout_qubits[-1], key='m{}'.format(level))

    @staticmethod
    def create_non_control_layer(final: bool, layer_qubits: List[cirq.Qid],
                                 symbols: Union[Iterable[sp.Symbol], sp.Symbol]) -> cirq.Circuit:
        circuit = cirq.Circuit()
        n = len(layer_qubits)
        for i, qubit in enumerate(layer_qubits):
            circuit.append(ModelCircuits.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3]))
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
            circuit.append(ModelCircuits.one_qubit_unitary(qubit, symbols[3 * i: 3 * i + 3], control_qubit, control))
        if not level == len(qubits) - 1:
            for i, (qubit_0, qubit_1) in enumerate(zip(layer_qubits, layer_qubits[1:] + [layer_qubits[0]])):
                circuit.append(
                    cirq.CNotPowGate(exponent=symbols[3 * n + i]).on(qubit_0, qubit_1).controlled_by(
                        control_qubit, control_values=[control]))
        return circuit

    @staticmethod
    def one_qubit_unitary(qubit: cirq.Qid, symbols: sp.Symbol, control_qubit: cirq.Qid = None,
                          control: int = None) -> cirq.Circuit:
        if control_qubit is None:
            yield cirq.X(qubit) ** symbols[0]
            yield cirq.Y(qubit) ** symbols[1]
            yield cirq.Z(qubit) ** symbols[2]
        else:
            yield cirq.XPowGate(exponent=symbols[0]).on(qubit).controlled_by(
                control_qubit, control_values=[control])
            yield cirq.YPowGate(exponent=symbols[1]).on(qubit).controlled_by(
                    control_qubit, control_values=[control])
            yield cirq.ZPowGate(exponent=symbols[2]).on(qubit).controlled_by(
                    control_qubit, control_values=[control])

    @staticmethod
    def ent_ops(qubits: 'List[cirq.Qid]'):
        yield (cirq.CNOT(q1, q2) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]]))

    @staticmethod
    def entangle_data_work(data: 'List[cirq.Qid]', work: 'List[cirq.Qid]'):
        yield  (cirq.CNOT(q0, q1) for q0, q1 in zip(data, work))

    @staticmethod
    def leakage(leaky_qubit: 'cirq.Qid', ancilla_qubit: 'cirq.Qid', symbol: 'sp.Symbol'):
        yield cirq.CNOT(control=ancilla_qubit, target=leaky_qubit)
        yield (cirq.Y ** symbol)(ancilla_qubit)
        yield cirq.CNOT(control=leaky_qubit, target=ancilla_qubit)
        yield ((cirq.Y ** symbol) ** -1)(ancilla_qubit)
        yield cirq.CNOT(control=leaky_qubit, target=ancilla_qubit)
        yield cirq.CNOT(control=ancilla_qubit, target=leaky_qubit)

    @staticmethod
    def leakage_circuit(leaky_qubits: 'List[cirq.Qid]', ancilla_qubits: 'List[cirq.Qid]',
                        symbols: 'Union[List[sp.Symbol], float]'):
        circuit = cirq.Circuit()
        for lq, aq, gamma in zip(leaky_qubits, ancilla_qubits, symbols):
            circuit += ModelCircuits.leakage(lq, aq, gamma)
        return circuit

    @staticmethod
    def quantum_OR(qubit_a: 'cirq.Qid', qubit_b: 'cirq.Qid', output_qubit: 'cirq.Qid'):
        sqrt_x = cirq.X ** 0.5

        yield sqrt_x(output_qubit)
        yield cirq.CNOT(control=qubit_a, target=output_qubit)

        yield sqrt_x(output_qubit)
        yield cirq.CNOT(control=qubit_b, target=output_qubit)

        yield (sqrt_x ** -1)(output_qubit)
        yield cirq.CNOT(control=qubit_a, target=output_qubit)

        yield (sqrt_x ** -1)(output_qubit)
        yield cirq.CNOT(control=qubit_b, target=output_qubit)

        yield cirq.CNOT(control=qubit_a, target=output_qubit)

    @staticmethod
    def OR_circuit(a_qubits: 'List[cirq.Qid]', b_qubits: 'List[cirq.Qid]', output_qubits: 'List[cirq.Qid]'):
        circuit = cirq.Circuit()
        for a, b, out in zip(a_qubits, b_qubits, output_qubits):
            circuit += ModelCircuits.quantum_OR(a, b, out)
        return circuit
