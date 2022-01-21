import itertools as it
import cirq
from typing import List, Tuple, Union, Iterable

from qubit_adapt.qubit_optimiser import QubitOptimiser


class VAE:

    ENTANGLING_GATE = cirq.ZZ

    def __init__(self,
                 measurements: List[Tuple[float, Union[Iterable[cirq.GateOperation], cirq.GateOperation]]],
                 simulator: cirq.Simulator = cirq.Simulator(), training_epochs: int = 1000, lr: float = 0.1,
                 optimiser: str = 'Adam'):
        self.measurements = measurements
        self.sim = simulator
        self.epochs = training_epochs
        self.lr = lr
        self.optimiser = optimiser
        self.qubits = self.get_qubits()
        self.n = len(self.qubits)

    def get_qubits(self):
        qubits = set()
        for coeff, ops in self.measurements:
            qubits.update(QubitOptimiser.measurement_qubits(ops))
        qubits = list(qubits)
        qubits.sort()
        return qubits

    def propose_large_ansatz(self, layers: int = 5) -> cirq.Circuit:
        variational_gates = QubitOptimiser.PARAMETERISED_GATES
        variational_layer = [g().on(q) for q in self.qubits for g in variational_gates]
        combinations = it.combinations(self.qubits, self.ENTANGLING_GATE.num_qubits())
        variational_layer.extend([self.ENTANGLING_GATE(*c) for c in combinations])
        proposed = cirq.Circuit()
        [proposed.append(variational_layer) for _ in range(layers)]
        return proposed

    def minimise_energy(self, circuit: cirq.Circuit, iterations: int = None):
        if iterations is None:
            iterations = int(self.epochs / 10)
        minimiser = QubitOptimiser(circuit, self.measurements, simulator=self.sim, lr=self.lr, optimiser=self.optimiser)
        for i in range(iterations):
            circuit = minimiser.optimise_circuit(circuit)
        energy = minimiser.energy_fn(circuit)
        return energy, circuit

    def return_gradients(self, large: cirq.Circuit, proposed: cirq.Circuit):
        comm_pos = cirq.Circuit()
        comm_neg = cirq.Circuit()
        comm_pos.append(large)
        comm_pos.append(proposed)
        comm_neg.append(proposed)
        comm_neg.append(large)
        minimiser = QubitOptimiser(comm_pos, self.measurements, simulator=self.sim, lr=self.lr, optimiser=self.optimiser)
        energy_pos = minimiser.energy_fn(comm_pos)
        energy_neg = minimiser.energy_fn(comm_neg)
        difference = energy_pos - energy_neg
        return difference

    def compare_ansatze(self, large: cirq.Circuit, proposed: cirq.Circuit):
        comp = cirq.Circuit()
        comp.append(large)
        comp.append(proposed)
        wf = self.sim.simulate(comp).final_state
        difference = 1 - wf[0]
        return difference
