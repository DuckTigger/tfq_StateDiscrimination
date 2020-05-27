import unittest
from qutrit_model import QutritModel


class TestQutritModel(unittest.TestCase):
    def test_circuit(self):
        model = QutritModel(4, 0.01)
        circuit = model.qutrit_discrimination_circuit()
        print(circuit)

    def test_create_model(self):
        model = QutritModel(4, 0.01)
        qutrit_model = model.qutrit_model()
        print(qutrit_model)



if __name__ == '__main__':
    unittest.main()
