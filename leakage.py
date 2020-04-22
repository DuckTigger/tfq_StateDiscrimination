import sympy as sp
import tensorflow_quantum as tfq
import cirq

from typing import List
from model_circuits import ModelCircuits


class LeakageModels:

    def __init__(self, n_work_qubits: int, n_data_qubits: int = 2):
        total_qubits = (3 * n_work_qubits) + n_data_qubits
        self.qubits = cirq.GridQubit.rect(int(total_qubits / 2), 2)
        self.readout_qubits = self.qubits[:n_work_qubits]
        self.ancilla_qubits = self.qubits[n_work_qubits: 2 * n_work_qubits]
        self.work_qubits = self.qubits[2 * n_work_qubits:3 * n_work_qubits]
        self.data_qubits = self.qubits[-n_data_qubits:]
        self.work_data_qubits = self.qubits[2 * n_work_qubits:]
        self.n = n_work_qubits + n_data_qubits

    def discrimination_circuit(self):
        output = cirq.Circuit()
        for i in range(len(self.work_qubits)):
            disc_symbols = sp.symbols('layer{}_0:{}'.format(i, 4 * self.n - i))
            leak_symbols = sp.symbols('lekage{}_0:{}'.format(i, self.n - i))
            output.append(ModelCircuits.create_layers(self.work_data_qubits, disc_symbols, i))
            output.append()


