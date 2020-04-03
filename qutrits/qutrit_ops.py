import cirq
import numpy as np


class QutritPlusGate(cirq.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return '[+1]'


class QutritMinusGate(cirq.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return '[-1]'


class QutritZPlusGate(cirq.SingleQubitGate):
    def _qid_shape(self):
        return (3,)

    def _unitary_(self):
        return np.array([[1, 0, 0],
                        [0, np.exp(2j * np.pi / 3), 0],
                        [0, 0, np.exp(4j * np.pi / 3)]])

    def _circuit_diagram_info(self, args):
        return '[+Z]'


class QutritZMinusGate(cirq.SingleQubitGate):
    def _qid_shape(self):
        return (3,)

    def _unitary_(self):
        return np.array([[1, 0, 0],
                         [0, np.exp(4j * np.pi / 3), 0],
                         [0, 0, np.exp(2j * np.pi / 3)]])

    def _circuit_diagram_info(self, args):
        return '[-Z]'


class C2QutritPlusGate(cirq.TwoQubitGate):
    """
    2-controlled Qutrit Plus Gate on two qutrits
    """
    def _qid_shape_(self):
        return (3,3,)

    def _unitary_(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=('2', '[+1]'))

class C2QutritQubitXGate(cirq.TwoQubitGate):
    """
    2-controlled Qutrit-Qubit CNOT gate
    """
    def _qid_shape_(self):
        return (3,2,)

    def _unitary_(self):
        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=('2', '[X]'))


q0 = cirq.LineQid(0, dimension=3)
q1 = cirq.LineQid(1, dimension=2)
qubits = [q0, q1]
circuit = cirq.Circuit(
    QutritPlusGate().on(q0),
    QutritPlusGate().on(q0),
    C2QutritQubitXGate().on(q0, q1),
    cirq.measure(*qubits, key='result')
)
print(circuit)

#circuit.append(cirq.measure(*q0, 'm'))
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1)
print(result)
