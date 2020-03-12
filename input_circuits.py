import cirq
from scipy.stats import truncnorm
import numpy as np


class InputCircuits:

    def __init__(self, n_qubits: int):
        self.qubits = cirq.GridQubit.rect(int(n_qubits/2), 2)
        self.n = n_qubits

    def create_discrimination_circuits(self, total_states: int = 1000, prop_a: float = 0.5,
                                     mu_a: float = 0.5, mu_b: float = 0.5, sigma_a: float = 0.01,
                                     sigma_b: float = 0.01):
        a_dist = truncnorm.rvs((0 - mu_a) / sigma_a, (1 - mu_a) / sigma_a, mu_a, sigma_a,
                               size=int(total_states * prop_a * 2) + 2)
        b_dist = truncnorm.rvs((0 - mu_b) / sigma_b, (1 - mu_b) / sigma_b, mu_b, sigma_b,
                               size=int(total_states * (1 - prop_a)) + 2)
        a_circuits = []
        b_circuits = []
        for i in range(int(total_states * prop_a * 2)):
            a_circuits.append(self.create_a(a_dist[i]))
        for i in range(int(total_states * (1 - prop_a))):
            b_circuits.append(self.create_b(b_dist[i]))
        return a_circuits, b_circuits

    def create_a(self, a: float):
        ops = [cirq.ry(2 * np.arcsin(a))(self.qubits[0])]
        ops.extend([cirq.Z(q) for q in self.qubits[1:]])
        circuit = cirq.Circuit(ops)
        return circuit

    def create_b(self, b: float):
        ops1 = [cirq.X(self.qubits[0]),
                cirq.ISWAP(self.qubits[0], self.qubits[1])**(2 * np.arcsin(b) / np.pi)]
        ops1.extend([cirq.Z(q) for q in self.qubits[2:]])
        circuit1 = cirq.Circuit(ops1)
        ops2 = [cirq.X(self.qubits[0]),
                cirq.ISWAP(self.qubits[0], self.qubits[1]) ** (- 2 * np.arcsin(b) / np.pi)]
        ops2.extend(cirq.Z(q) for q in self.qubits[2:])
        circuit2 = cirq.Circuit(ops2)
        return circuit1, circuit2
