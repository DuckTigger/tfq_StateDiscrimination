import unittest
import cirq
import numpy

from input_circuits import InputCircuits


class TestRandomCircuits(unittest.TestCase):
    def test_circuits_random(self):
        circuit_creator = InputCircuits(4)
        circuits, circuit_creator.create_random_circuits(100)



if __name__ == '__main__':
    unittest.main()
