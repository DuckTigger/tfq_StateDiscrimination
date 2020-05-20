import unittest
import cirq
import numpy as np

from model_circuits import ModelCircuits


class TestModelCircuits(unittest.TestCase):

    def test_qutrit_CSUM(self):
        n =4
        q = cirq.LineQubit.range(n)
        c = cirq.Circuit(ModelCircuits.qutrit_CSUM(q[1], q[0], q[3], q[2], 1.))

        c.append([cirq.measure(q_, key='m') for q_ in q], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        test = c.unitary().astype(int)
        true = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        np.testing.assert_array_equal(test, true)
        
    def test_qutrit_CX(self):
        n = 3
        q = cirq.LineQubit.range(n)
        c = cirq.Circuit(ModelCircuits.qutrit_CX(q[0], q[2], q[1], 1.))
        c.append([cirq.measure(q_, key='m') for q_ in q], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        test =c.unitary().astype(int)
        true = [[1, 0, 0 ,0, 0, 0, 0, 0],
                [0 ,1 ,0, 0,0, 0 ,0 ,0],
                [0, 0, 1, 0, 0 ,0 ,0 ,0],
                [0, 0 ,0 ,1 ,0 ,0 ,0 ,0],
                [0, 0, 0, 0, 0, 0, 1, 0,],
                [0, 0, 0 ,0 ,1 ,0 ,0 ,0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0 ,0 ,0 ,0, 1]]
        np.testing.assert_array_equal(test, true)

    def test_toffoli(self):
        n = 3
        q = cirq.LineQubit.range(n)
        c = cirq.Circuit(ModelCircuits.toffoli(q[0], q[1], q[2], 1.))
        c.append([cirq.measure(q_, key='m') for q_ in q], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        test = c.unitary().astype(int)
        true = [[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0]]
        np.testing.assert_array_equal(test, true)

    def test_qutrit_CX2(self):
        n = 3
        q = cirq.LineQubit.range(n)
        c = cirq.Circuit(ModelCircuits.qutrit_CX2(q[0], q[2], q[1], 1.))
        c.append([cirq.measure(q_, key='m') for q_ in q], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        test =c.unitary().astype(int)
        true = [[1, 0, 0 ,0, 0, 0, 0, 0],
                [0 ,1 ,0, 0, 0, 0 ,0 ,0],
                [0, 0, 1, 0, 0, 0 ,0 ,0],
                [0, 0 ,0 ,1 ,0 ,0 ,0 ,0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0 ,0 ,0 ,0 ,1 ,0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0 ,0 ,0 ,0, 1]]
        np.testing.assert_array_equal(test, true)

    def test_qutrit_X(self):
        n = 2
        q = cirq.LineQubit.range(n)
        c = cirq.Circuit(ModelCircuits.qutrit_X(q[1], q[0], 1.))
        test = c.unitary().astype(int)
        true = [[0, 1, 0, 0], [0, 0, 1, 0,], [1, 0,0, 0,], [0, 0, 0, 1]]
        np.testing.assert_array_equal(test, true)

    def test_qutrit_X2(self):
        n = 2
        q = cirq.LineQubit.range(n)
        c = cirq.Circuit(ModelCircuits.qutrit_X2(q[1], q[0], 1.))
        test = c.unitary().astype(int)
        true = [[0, 0, 1, 0], [1, 0, 0, 0,], [0, 1, 0, 0,], [0, 0, 0, 1]]
        np.testing.assert_array_equal(test, true)

    def test_toffoli_exponent(self):
        n = 3
        q = cirq.LineQubit.range(n)
        exp = 0.5
        c_test = np.round(cirq.Circuit(ModelCircuits.toffoli(q[0], q[1], q[2], exp)).unitary(), 1)
        c_true = cirq.Circuit(cirq.TOFFOLI(q[0], q[1], q[2]) ** exp).unitary()
        print(c_test, '\n')
        print(c_true)
        np.testing.assert_array_equal(c_test, c_true)


    @staticmethod
    def bin_list_to_num(l):
        s = ''.join(str(x) for x in l)
        return int(s, 2)

    @staticmethod
    def num_to_bin_list(n, dim=4):
        s = format(n, '0{}b'.format(dim))
        l = []
        l.extend(int(i) for i in s)
        return l


if __name__ == '__main__':
    unittest.main()
