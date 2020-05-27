import unittest
import cirq
import numpy as np

from qutrits.qutrit_ops import QutritPlusGate, QutritZPlusGate, QutritMinusGate, QutritZMinusGate, C2QutritPlusGate,\
    C2QutritQubitXGate
from qutrits.qutrit_noise import TwoQutritDepolarizingChannel, SingleQutritDepolarizingChannel,\
    two_qutrit_depolarize, qutrit_depolarise

from circuit_layers import SubCircuits

class TestQutritOps(unittest.TestCase):
    def test_qutrit_circuit(self):
        qutrits = [cirq.LineQid(0, dimension=3), cirq.LineQid(1, dimension=3)]
        circuit = cirq.Circuit(
            QutritZPlusGate().on(qutrits[0]),
            QutritZMinusGate().on(qutrits[0]),
            QutritPlusGate().on(qutrits[1]),
            QutritMinusGate().on(qutrits[1]),
            cirq.measure(*qutrits, key='m')
        )
        sim = cirq.Simulator()
        res = sim.run(circuit, repetitions=10)
        for meas in res.measurements['m']:
            np.testing.assert_array_equal(meas, [0,0])


class TestCircuitModelQutrits(unittest.TestCase):

    def test_qtX(self):
        target = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        q = cirq.LineQubit.range(2)
        u = cirq.Circuit([SubCircuits.qutrit_X(q[0], q[1])]).unitary()
        np.testing.assert_array_equal(u, target)

    def test_qtX2(self):
        target = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=complex)
        q = cirq.LineQubit.range(2)
        u = cirq.Circuit([SubCircuits.qutrit_X2(q[0], q[1])]).unitary()
        np.testing.assert_array_equal(u, target)

    def test_qtZ(self):
        target = np.array(
            [[1, 0, 0, 0], [0, np.exp(2*np.pi*1j/3), 0, 0], [0, 0, np.exp(4*np.pi*1j/3), 0], [0, 0, 0, 1]],
            dtype=complex)
        q = cirq.LineQubit.range(2)
        u = cirq.Circuit([SubCircuits.qutrit_Z(q[0], q[1])]).unitary()
        np.testing.assert_array_equal(u, target)

    def test_qtZ2(self):
        target = np.array(
            [[1, 0, 0, 0], [0, np.exp(4*np.pi*1j/3), 0, 0], [0, 0, np.exp(2*np.pi*1j/3), 0], [0, 0, 0, 1]],
            dtype=complex)
        q = cirq.LineQubit.range(2)
        u = cirq.Circuit([SubCircuits.qutrit_Z2(q[0], q[1])]).unitary()
        np.testing.assert_array_equal(u, target)


if __name__ == '__main__':
    unittest.main()
