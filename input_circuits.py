import cirq
from scipy.stats import truncnorm
import tensorflow_quantum as tfq
import numpy as np

from typing import List


class InputCircuits:

    def __init__(self, n_qubits: int):
        self.qubits = cirq.GridQubit.rect(int(n_qubits/2), 2)
        self.n = n_qubits

    def create_discrimination_circuits(self, total_states: int = 1000,
                                       mu_a: float = 0.5, mu_b: float = 0.5, sigma_a: float = 0.01,
                                       sigma_b: float = 0.01):
        circuits, labels = self.discrimination_circuits_labels(total_states, mu_a, mu_b, sigma_a, sigma_b)
        return self.return_tensors(circuits, labels)

    def create_random_circuits(self, total_states: int = 1000, lower: int = 0, upper: int = 1,
                               mu_a: float = 0.5, sigma_a: float = 0.15):
        circuits, labels = self.random_circuits_labels(total_states, lower, upper, mu_a, sigma_a)
        return self.return_tensors(circuits, labels)

    def discrimination_circuits_labels(self, total_states: int = 1000,
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
                sign = np.random.choice([0, 1])
                circuits.append(self.create_b(b_dist[i], sign))
        return circuits, labels

    def random_circuits_labels(self, total_states: int = 1000, lower: int = 0, upper: int = 1,
                               mu_a: float = 0.5, sigma_a: float = 0.15):
        dist = truncnorm.rvs((lower - mu_a) / sigma_a, (upper - mu_a) / sigma_a, mu_a, sigma_a,
                             size=4 * total_states + 2)
        circuits = []
        labels = []
        for i in range(0, total_states, 4):
            label = np.random.choice([0, 1])
            labels.append(label)
            if label:
                circuits.append(self.create_random(dist[i: i + 2], b=False))
            else:
                circuits.append(self.create_random(dist[i + 2:i + 4], b=True))
        return circuits, labels

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

    def create_random(self, rotations: np.array, b: bool):
        ops = [cirq.ry(2 * np.arcsin(rotations[0]))(self.qubits[0])**(-1j if b else 1),
               cirq.ry(2 * np.arcsin(rotations[1]))(self.qubits[1])**(-1j if b else 1)]
        ops.extend([cirq.Z(q) for q in self.qubits[2:]])
        circuit = cirq.Circuit(ops)
        return circuit

    def create_random_circuits(self, total_states: int = 1000, lower: int = 0, upper: int = 1,
                               mu_a: float = 0.5, sigma_a: float = 0.15):
        circuits, labels = self.random_circuits_labels(total_states, lower, upper, mu_a, sigma_a)
        return self.return_tensors(circuits, labels)


    @staticmethod
    def return_tensors(circuits: List, labels: List):
        split = int(len(circuits) * 0.7)
        train_circuits = circuits[:split]
        test_circuits = circuits[split:]

        train_labels = labels[:split]
        test_labels = labels[split:]

        return tfq.convert_to_tensor(train_circuits), np.array(train_labels), \
               tfq.convert_to_tensor(test_circuits), np.array(test_labels)