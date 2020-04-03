import cirq
import cmath
from cirq import value, protocols
# noinspection PyProtectedMember
from cirq._compat import proper_repr
from cirq.ops import gate_features, eigen_gate, raw_types
from typing import Sequence, Tuple

import numpy as np
import itertools as it

def create_noise_matrices():
    tr_x = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    tr_z = np.array([[1, 0, 0], [0, np.exp(2j * np.pi / 3), 0], [0, 0, np.exp(4j * np.pi / 3)]])
    eye = np.eye(3)
    z_sq  = tr_z @ tr_z
    x_sq = tr_x @ tr_x

    x_set = [eye, tr_x, x_sq]
    z_set = [eye, tr_z, z_sq]
    prods = [a @ b for a, b in it.product(x_set, z_set)]
    return prods

def two_q_noise_matrices():
    one_q = create_noise_matrices()
    mats = [np.kron(a, b) for a, b in it.product(one_q, one_q)]
    return mats

@cirq.value.value_equality
class SingleQutritDepolarizingChannel(gate_features.SingleQubitGate):

    def __init__(self, p):
        self.unitary_matrices = create_noise_matrices()
        self._p = p
        self._p_i = 1 - 8 * p

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        iden = [(self._p_i, self.unitary_matrices[0])]
        iden.extend([(self.p, m) for m in self.unitary_matrices[1:]])
        return tuple(iden)

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values(self):
        return self._p

    def __repr__(self) -> str:
        return 'qutrit_depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'qutrit_depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> str:
        return '3+D({!r})'.format(self._p)


def qutrit_depolarise(p: float) -> SingleQutritDepolarizingChannel:
    return SingleQutritDepolarizingChannel(p)

@cirq.value.value_equality
class TwoQutritDepolarizingChannel(gate_features.TwoQubitGate):

    def __init__(self, p: float):
        self.unitary_matrices = two_q_noise_matrices()
        self._p = p
        self._p_i = 1 - 80 * p

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        iden = [(self._p_i, self.unitary_matrices[0])]
        iden.extend([(self.p, m) for m in self.unitary_matrices[1:]])
        return tuple(iden)

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values(self):
        return self._p

    def __repr__(self) -> str:
        return 'two_qutrit_depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'two_qutrit_depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> str:
        return '2*3+D({!r})'.format(self._p)


def two_qutrit_depolarize(p: float) -> TwoQutritDepolarizingChannel:
    return TwoQutritDepolarizingChannel(p)


if __name__ == '__main__':
    print(two_q_noise_matrices())
