import cirq
from scipy.stats import truncnorm
import tensorflow_quantum as tfq
import numpy as np


class InputCircuits:

    def __init__(self, n_qubits: int):
        self.qubits = cirq.GridQubit.rect(int(n_qubits/2), 2)
        self.n = n_qubits

    def create_discrimination_circuits(self, total_states: int = 1000, prop_a: float = 0.5,
                                     mu_a: float = 0.5, mu_b: float = 0.5, sigma_a: float = 0.01,
                                     sigma_b: float = 0.01):
        a_dist = truncnorm.rvs((0 - mu_a) / sigma_a, (1 - mu_a) / sigma_a, mu_a, sigma_a,
                               size=total_states)
        b_dist = truncnorm.rvs((0 - mu_b) / sigma_b, (1 - mu_b) / sigma_b, mu_b, sigma_b,
                               size=total_states)
        circuits = []
        labels = []
        for i in range(total_states):
            label = np.random.choice([0, 1])
            labels.append(label)
            if label:
                circuits.append(self.create_a(a_dist[i]))
            else:
                sign = np.random.choice([0,1])
                circuits.append(self.create_b(b_dist[i], sign))

        split = int(len(circuits) * 0.7)
        train_circuits = circuits[:split]
        test_circuits = circuits[split:]

        train_labels = labels[:split]
        test_labels = labels[split:]

        return tfq.convert_to_tensor(train_circuits), np.array(train_labels), \
               tfq.convert_to_tensor(test_circuits), np.array(test_labels)

    def create_a(self, a: float):
        ops = [cirq.ry(2 * np.arcsin(a))(self.qubits[0])]
        ops.extend([cirq.Z(q) for q in self.qubits[1:]])
        circuit = cirq.Circuit(ops)
        return circuit

    def create_b(self, b: float, sign: int):
        ops = [cirq.X(self.qubits[0]),
                cirq.ISWAP(self.qubits[0], self.qubits[1])**((-1 ** sign) * 2 * np.arcsin(b) / np.pi)]
        ops.extend([cirq.Z(q) for q in self.qubits[2:]])
        circuit = cirq.Circuit(ops)
        return circuit
