import cirq
import tensorflow as tf
import itertools as it
import numpy as np
from typing import List, Tuple, Union, Iterable

from adapt_base.adapt_base import QubitOptimiser
from qVAE.qVAE import VAE


class Reconstructor:

    def __init__(self, measurements: List[Tuple[float, Union[Iterable[cirq.GateOperation], cirq.GateOperation]]],
                 simulator: cirq.Simulator = cirq.Simulator(), training_epochs: int = 1000, lr: float = 0.1):
        self.measurements = measurements
        self.sim = simulator
        self.epochs = training_epochs
        self.lr = lr
        self.qubits = self.get_qubits()
        self.allowed_ops = self.get_allowed_ops()
        self.n = len(self.qubits)

    def get_allowed_ops(self) -> List[cirq.GateOperation]:
        param_gates = QubitOptimiser.PARAMETERISED_GATES
        entangling = VAE.ENTANGLING_GATE
        ops = []
        for qubit in self.qubits:
            for gate in param_gates:
                ops.extend([gate().on(qubit)])

        combinations = it.combinations(self.qubits, entangling.num_qubits())
        for combo in combinations:
            ops.extend([entangling(*combo)])
        return ops

    def get_qubits(self) -> List[cirq.Qid]:
        qubits = set()
        for coeff, ops in self.measurements:
            qubits.update(QubitOptimiser.measurement_qubits(ops))
        qubits = list(qubits)
        qubits.sort()
        return qubits

    def reconstruct_circuit(self):
        pass

    def tensor_to_circuit(self, input_tensor: tf.Tensor) -> cirq.Circuit:
        max_ops = len(self.allowed_ops)
        gate_choice = tf.cast(tf.multiply(tf.pow(input_tensor[0], 1), max_ops), tf.int32)
        power = tf.multiply(input_tensor[1], 2 * np.pi)

        output_circuit = cirq.Circuit()
        for i in range(len(gate_choice)):
            gate = self.allowed_ops[int(gate_choice[i].numpy())]
            output_circuit.append(gate ** power[i].numpy())
        return output_circuit