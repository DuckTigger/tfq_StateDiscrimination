import cirq
import numpy as np
from pytket.backends.ibm import IBMQBackend, AerBackend
from typing import List, Dict
from cirq import study, protocols
from cirq.sim.simulator import _verify_unique_measurement_keys

from cirq_convert import cirq_to_tk


class QCBackend(cirq.Sampler):

    def __init__(self,
                 backend_name: str = None
                 ):
        if backend_name is None:
            self.backend = AerBackend()
        else:
            self.backend = IBMQBackend(backend_name)

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
        measurements = {'out': np.array([], dtype=np.uint8)}
        to_tket = cirq_to_tk(resolved)
        self.backend.compile_circuit(to_tket)
        res = self.backend.process_circuit(to_tket, repetitions)
        shots = self.backend.get_shots(res)
        for measure in shots:
            measurements['out'] = np.append(measurements['out'], measure)

        return measurements


if __name__ == '__main__':
     to_qc = QCBackend('ibmq_essex')
     q = cirq.LineQubit.range(2)
     circuit = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(*q), cirq.measure(q[0], key='m0'), cirq.measure(q[1], key='m1')])
     print(to_qc.run_sweep(circuit, {}, 10))
