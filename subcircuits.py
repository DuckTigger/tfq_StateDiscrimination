import cirq
import sympy as sp
import itertools as it
from typing import List, Iterable, Union


class SubCircuits:

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
    def qutrit_unitary(qubit: 'cirq.Qid', ancilla: 'cirq.Qid', symbols: 'Iterable[sp.Symbol]'):
        x = (None, SubCircuits.qutrit_X, SubCircuits.qutrit_X2)
        z = (None, SubCircuits.qutrit_Z, SubCircuits.qutrit_Z2)
        prod = filter(lambda x: x.count(SubCircuits.qutrit_I) != 2, it.product(x, z))
        for gates, symbol in zip(prod, symbols):
            for gate in gates:
                if gate is not None:
                    yield gate(ancilla, qubit, symbol)

    @staticmethod
    def ent_ops(qubits: 'List[cirq.Qid]'):
        yield (cirq.CNOT(q1, q2) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]]))

    @staticmethod
    def entangle_data_work(data: 'List[cirq.Qid]', work: 'List[cirq.Qid]'):
        yield (cirq.CNOT(q0, q1) for q0, q1 in zip(data, work))

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
            circuit += SubCircuits.leakage(lq, aq, gamma)
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
            circuit += SubCircuits.quantum_OR(a, b, out)
        return circuit

    @staticmethod
    def qutrit_CSUM(control_ancilla: 'cirq.Qid', control_qubit: 'cirq.Qid',
                    target_ancilla: 'cirq.Qid', target_qubit: 'cirq.Qid', symbol: 'sp.Symbol'):
        yield SubCircuits.qutrit_CX(control_ancilla, target_ancilla, target_qubit, symbol)
        yield SubCircuits.qutrit_CX2(control_qubit, target_ancilla, target_qubit, symbol)

    @staticmethod
    def qutrit_CX(control: 'cirq.Qid', target_qubit: 'cirq.Qid', target_ancilla: 'cirq.Qid', symbol: 'sp.Symbol'):
        yield cirq.X(target_ancilla)
        yield SubCircuits.toffoli(control, target_ancilla, target_qubit, symbol)
        yield cirq.X(target_ancilla)
        yield cirq.X(target_qubit)
        yield SubCircuits.toffoli(control, target_qubit, target_ancilla, symbol)
        yield cirq.X(target_qubit)

    @staticmethod
    def qutrit_CX2(control: 'cirq.Qid', target_qubit: 'cirq.Qid', target_ancilla: 'cirq.Qid', symbol: 'sp.Symbol'):
        yield cirq.X(target_qubit)
        yield SubCircuits.toffoli(control, target_qubit, target_ancilla, symbol)
        yield cirq.X(target_qubit)
        yield cirq.X(target_ancilla)
        yield SubCircuits.toffoli(control, target_ancilla, target_qubit, symbol)
        yield cirq.X(target_ancilla)

    @staticmethod
    def toffoli(control_0: 'cirq.Qid', control_1: 'cirq.Qid', target: 'cirq.Qid',
                exponent: Union[float, sp.Symbol] = 1.):
        # Taken from the cirq decomposition
        p = cirq.T ** exponent
        yield cirq.H(target)
        yield p(control_0)
        yield p(control_1)
        yield p(target)
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield p(control_1) ** -1
        yield p(target)
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield p(target) ** -1
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield p(target) ** -1
        yield cirq.CNOT(control_0, control_1)
        yield cirq.CNOT(control_1, target)
        yield cirq.H(target)

    @staticmethod
    def qutrit_I(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        pass

    @staticmethod
    def qutrit_X2(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        yield cirq.X(qubit)
        yield cirq.CNOT(qubit, ancilla) ** exponent
        yield cirq.X(qubit)
        yield cirq.X(ancilla)
        yield cirq.CNOT(ancilla, qubit) ** exponent
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
        yield cirq.CZ(qubit, ancilla) ** (4 * exponent / 3)
        yield cirq.X(qubit)
        yield cirq.X(ancilla)
        yield cirq.CZ(ancilla, qubit) ** (2 * exponent / 3)
        yield cirq.X(ancilla)

    @staticmethod
    def qutrit_Z2(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', exponent: float = 1.):
        yield cirq.X(qubit)
        yield cirq.CZ(qubit, ancilla) ** (2 * exponent / 3)
        yield cirq.X(qubit)
        yield cirq.X(ancilla)
        yield cirq.CZ(ancilla, qubit) ** (4 * exponent / 3)
        yield cirq.X(ancilla)

    @staticmethod
    def qutrit_CNOT(ancilla: 'cirq.Qid', qubit: 'cirq.Qid', target: 'cirq.Qid', exponent: float = 1.):
        yield cirq.CNOT(qubit, target) ** exponent
        yield cirq.CNOT(ancilla, target) ** exponent