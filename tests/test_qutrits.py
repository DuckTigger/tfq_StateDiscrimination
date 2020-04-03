import unittest
import cirq
import numpy as np

from qutrits.qutrit_ops import QutritPlusGate, QutritZPlusGate, QutritMinusGate, QutritZMinusGate, C2QutritPlusGate,\
    C2QutritQubitXGate
from qutrits.qutrit_noise import TwoQutritDepolarizingChannel, SingleQutritDepolarizingChannel,\
    two_qutrit_depolarize, qutrit_depolarise


class MyTestCase(unittest.TestCase):
    def test_qutrit_circuit(self):
        qutrits = [cirq.GridQubit(0, 0, dimension=3), cirq.GridQubit(0, 1, dimension=3)]
        circuit = cirq.Circuit()


if __name__ == '__main__':
    unittest.main()
