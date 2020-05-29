import cirq
import pytket
import qiskit
import numpy as np
from pytket.backends.ibm import IBMQBackend, AerBackend
import tensorflow_quantum as tfq
from typing import List, Dict
from cirq import study, protocols
from cirq.sim.simulator import _verify_unique_measurement_keys

from cirq_convert import cirq_to_tk


class QCBackend(cirq.Sampler):

    def init(self):
        self.backend = AerBackend()

    def run_sweep(
            self,
            program: 'cirq.Circuit',
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:

        if not program.has_measurements():
            raise ValueError("Circuit has no measurements to sample")

        _verify_unique_measurement_keys(program)
        trial_results = []
        for param_resolver in study.to_resolvers(params):
            measurements = self.convert_and_run(circuit=program,
                                                param_resolver=param_resolver,
                                                repetitions=repetitions)
            trial_results.append(
                study.TrialResult.from_single_parameter_set(params=param_resolver,
                                                        measurements=measurements))

        return trial_results

    def convert_and_run(self, circuit: cirq.Circuit,
                        param_resolver: cirq.ParamResolver,
                        repetitions: int
                            ) -> Dict[str, np.ndarray]:
        resolved = protocols.resolve_parameters(circuit, param_resolver)
        measurement_keys = circuit.all_measurement_keys()
        to_tket = cirq_to_tk(resolved)
        res = self.backend.get_shots(to_tket, repetitions)


