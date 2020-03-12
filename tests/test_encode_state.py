import unittest
import cirq
import numpy as np
# import tensorflow as tf
# import tensorflow_quantum as tfq
import sympy as sy

from encode_state import EncodeState


class TestEncodeState(unittest.TestCase):
    def test_encoding_layers(self):
        n = 6
        encoder = EncodeState(n)
        test_circuit = encoder.create_encoding_layers()
        print(test_circuit)
        self.assertEqual(len(test_circuit), (3 * (n + 1)))

    def test_one_qubit_unitary(self):
        sym = sy.symbols('0:3')
        qubit = cirq.LineQubit.range(1)[0]
        unit = EncodeState.one_qubit_unitary(qubit, sym)
        circuit = cirq.Circuit(unit)
        thetas = [np.random.rand() for _ in range(3)]
        test = cirq.Circuit([cirq.X(qubit)**thetas[0], cirq.Y(qubit)**thetas[1], cirq.Z(qubit)**thetas[2]])
        sim = cirq.Simulator()
        resolve = cirq.ParamResolver({'0': float(thetas[0]), '1': thetas[1], '2': thetas[2]})
        np.testing.assert_array_almost_equal(sim.simulate(circuit, param_resolver=resolve).final_state, test.final_wavefunction())

    def test_encode_state(self):
        n = 6
        state = cirq.testing.random_superposition(2**n)
        encoder = EncodeState(n)
        encoder.encode_state(state)



if __name__ == '__main__':
    unittest.main()
