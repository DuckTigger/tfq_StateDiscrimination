import cirq
import sympy as sp
import itertools as it
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
    def create_qutrit_layers(work_qubits: 'List[cirq.Qid]', ancilla_qubits: 'List[cirq.Qid]',
                            readout_qubits: 'List[cirq.Qid]',
                            level: int, layer_qubits: 'List[cirq.Qid]', symbols:sp.Symbol) -> 'cirq.Circuit':
        circuit = cirq.Circuit()
        n = len(layer_qubits)
                                                                                                    
        for i, (qubit, ancilla) in enumerate(zip(work_qubits, ancilla_qubits)):
            pass

    @staticmethod
    def create_qutrit_layer(layer_qubits: 'List[cirq.Qid]', layer_ancilla: 'List[cirq.Qid]',
                            final: bool, symbols: sp.Symbol) -> 'cirq.Circuit':
        n = len(layer_qubits)
        for i, (qubit, ancilla) in enumerate(zip(layer_qubits, layer_ancilla)):
            yield ModelCircuits.qutrit_unitary(qubit, ancilla, symbols[8 * i: 8 * i + 8])
        if not final:
            for i, (control_qubit, control_ancilla,
                    target_qubit, target_ancilla) in enumerate(zip(layer_qubits, layer_ancilla,
                                                                   layer_qubits[1:] + layer_qubits[0],
                                                                   layer_ancilla[1:] + layer_ancilla[0])):
                pass


    @staticmethod
    def qutrit_unitary(qubit: 'cirq.Qid', ancilla: 'cirq.Qid', symbols: 'sp.Symbol'):
        x = (ModelCircuits.qutrit_I, ModelCircuits.qutrit_X, ModelCircuits.qutrit_X2)
        z = (ModelCircuits.qutrit_I, ModelCircuits.qutrit_Z, ModelCircuits.qutrit_Z2)
        prod = filter(lambda x: x.count(ModelCircuits.qutrit_I) != 2, it.product(x, z))
        for (gate1, gate2), symbol in zip(prod, symbols):
            yield gate1(ancilla, qubit, symbol)
            yield gate2(ancilla, qubit, symbol)

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

    @staticmethod
    def qutrit_CSUM(control_ancilla : 'cirq.Qid', control_qubit: 'cirq.Qid',
                    target_ancilla: 'cirq.Qid', target_qubit: 'cirq.Qid', symbol: 'sp.Symbol'):
        yield ModelCircuits.qutrit_CX(control_ancilla, target_ancilla, target_qubit, symbol)
        yield ModelCircuits.qutrit_CX2(control_qubit, target_ancilla, target_qubit, symbol)

    @staticmethod
    def qutrit_CX(control: 'cirq.Qid', target_qubit: 'cirq.Qid', target_ancilla: 'cirq.Qid', symbol: 'sp.Symbol'):
        yield cirq.X(target_ancilla)
        yield ModelCircuits.toffoli(control, target_ancilla, target_qubit, symbol)
        yield cirq.X(target_ancilla)
        yield cirq.X(target_qubit)
        yield ModelCircuits.toffoli(control, target_qubit, target_ancilla, symbol)
        yield cirq.X(target_qubit)

    @staticmethod
    def qutrit_CX2(control: 'cirq.Qid', target_qubit: 'cirq.Qid', target_ancilla: 'cirq.Qid', symbol: 'sp.Symbol'):
        yield cirq.X(target_qubit)
        yield ModelCircuits.toffoli(control,  target_qubit, target_ancilla, symbol)
        yield cirq.X(target_qubit)
        yield cirq.X(target_ancilla)
        yield ModelCircuits.toffoli(control, target_ancilla, target_qubit, symbol)
        yield cirq.X(target_ancilla)

    @staticmethod
    def toffoli(control_0: 'cirq.Qid', control_1: 'cirq.Qid', target: 'cirq.Qid',
                exponent: Union[float, sp.Symbol] = 1.):
        # Taken from the cirq decomposition
        p = cirq.T**exponent
        yield cirq.H(target)
        yield p(control_0)
        yield p(control_1)
        yield p(target)
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield p(control_1)**-1
        yield p(target)
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield p(target)**-1
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield p(target)**-1
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield cirq.H(target)

    @staticmethod
    def qutrit_I(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        yield cirq.identity_each(ancilla, qubit)

    @staticmethod
    def qutrit_X2(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        yield cirq.X(qubit)
        yield cirq.CNOT(qubit, ancilla)**exponent
        yield cirq.X(qubit)
        yield cirq.X(ancilla)
        yield cirq.CNOT(ancilla, qubit)**exponent
        yield cirq.X(ancilla)

    @staticmethod
    def qutrit_X(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        yield cirq.X(ancilla)
        yield cirq.CNOT(ancilla, qubit) ** exponent
        yield cirq.X(ancilla)
        yield cirq.X(qubit)
        yield cirq.CNOT(qubit, ancilla) ** exponent
        yield cirq.X(qubit)

    @staticmethod
    def qutrit_Z(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        yield cirq.X(qubit)
        yield cirq.CZ(qubit, ancilla) ** (4*exponent/3)
        yield cirq.X(qubit)
        yield cirq.X(ancilla)
        yield cirq.CZ(ancilla, qubit) ** (2*exponent / 3)
        yield cirq.X(ancilla)

    @staticmethod
    def qutrit_Z2(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        yield cirq.X(qubit)
        yield cirq.CZ(qubit, ancilla) ** (2*exponent/3)
        yield cirq.X(qubit)
        yield cirq.X(ancilla)
        yield cirq.CZ(ancilla, qubit) ** (4*exponent / 3)
        yield cirq.X(ancilla)

    @staticmethod
    def qutrit_CNOT(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', target: 'cirq.Qid', exponent: float = 1.):
        yield cirq.CNOT(qubit, target) ** exponent
        yield cirq.CNOT(ancilla, target) ** exponent
