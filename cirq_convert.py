# Copyright 2019-2020 Cambridge Quantum Computing
#
# Licensed under a Non-Commercial Use Software Licence (the "Licence");
# you may not use this file except in compliance with the Licence.
# You may obtain a copy of the Licence in the LICENCE file accompanying
# these documents or at:
#
#     https://cqcl.github.io/pytket/build/html/licence.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence, but note it is strictly for non-commercial use.

# One modification to allow for the use of phased CNOT gates

"""Methods to allow conversion between Cirq and t|ket> data types, including Circuits and Devices
"""

from typing import List, Generator, Dict, Union, Iterator
import cirq
from cirq.google import XmonDevice
from cirq import Qid, LineQubit, GridQubit
from cirq.ops import NamedQubit, measure

import pytket
Circuit, OpType, Qubit, Bit = pytket.Circuit, pytket.OpType, pytket.Qubit, pytket.Bit,

from sympy import pi, Expr

# For translating cirq circuits to tket circuits
cirq_common = cirq.ops.common_gates
cirq_pauli = cirq.ops.pauli_gates

# map cirq common gates to pytket gates
_cirq2ops_mapping = {
    cirq_common.CNOT: OpType.CX,
    cirq_common.H: OpType.H,
    cirq_common.MeasurementGate: OpType.Measure,
    cirq_common.XPowGate: OpType.Rx,
    cirq_common.YPowGate: OpType.Ry,
    cirq_common.ZPowGate: OpType.Rz,
    cirq_common.S: OpType.S,
    cirq_common.SWAP: OpType.SWAP,
    cirq_common.T: OpType.T,
    cirq_pauli.X: OpType.X,
    cirq_pauli.Y: OpType.Y,
    cirq_pauli.Z: OpType.Z,
    cirq_common.CZPowGate: OpType.CRz,
    cirq_common.CZ: OpType.CZ,
    cirq_common.ISwapPowGate: OpType.ISWAP,
    cirq.ops.parity_gates.ZZPowGate: OpType.ZZPhase,
    cirq.ops.parity_gates.XXPowGate: OpType.XXPhase,
    cirq.ops.parity_gates.YYPowGate: OpType.YYPhase,
    cirq.ops.PhasedXPowGate: OpType.PhasedX,
}
# reverse mapping for convenience
_ops2cirq_mapping = dict((reversed(item) for item in _cirq2ops_mapping.items()))
# _ops2cirq_mapping[OpType.X] = cirq_pauli.X
# _ops2cirq_mapping[OpType.Y] = cirq_pauli.Y
# _ops2cirq_mapping[OpType.Z] = cirq_pauli.Z
# spot special rotation gates
_constant_gates = (
    cirq_common.CNOT,
    cirq_common.H,
    cirq_common.S,
    cirq_common.SWAP,
    cirq_common.T,
    cirq_pauli.X,
    cirq_pauli.Y,
    cirq_pauli.Z,
    cirq_common.CZ,
)
_rotation_types = (
    cirq_common.XPowGate,
    cirq_common.YPowGate,
    cirq_common.ZPowGate,
    cirq_common.CZPowGate,
    cirq_common.ISwapPowGate,
    cirq.ops.parity_gates.ZZPowGate,
    cirq.ops.parity_gates.XXPowGate,
    cirq.ops.parity_gates.YYPowGate,
)


def cirq_to_tk(circuit: cirq.Circuit) -> Circuit:
    """Converts a Cirq :py:class:`Circuit` to a :math:`\\mathrm{t|ket}\\rangle` :py:class:`Circuit` object.

       :param circuit: The input Cirq :py:class:`Circuit`

       :raises NotImplementedError: If the input contains a Cirq :py:class:`Circuit` operation which is not yet supported by pytket

       :return: The :math:`\\mathrm{t|ket}\\rangle` :py:class:`Circuit` corresponding to the input circuit
    """
    tkcirc = Circuit()
    qmap = {}
    for qb in circuit.all_qubits():
        if isinstance(qb, LineQubit):
            id = Qubit("q", qb.x)
        elif isinstance(qb, GridQubit):
            id = Qubit("g", qb.row, qb.col)
        elif isinstance(qb, NamedQubit):
            id = Qubit(qb.name)
        else:
            raise NotImplementedError("Cannot convert qubits of type " + str(type(qb)))
        tkcirc.add_qubit(id)
        qmap.update({qb: id})
    for moment in circuit:
        for op in moment.operations:
            gate = op.gate
            gatetype = type(gate)

            qb_lst = [qmap[q] for q in op.qubits]

            n_qubits = len(op.qubits)

            if gatetype == cirq_common.HPowGate and gate.exponent == 1:
                gate = cirq_common.H
            elif gatetype == cirq_common.CNotPowGate and gate.exponent == 1:
                gate = cirq_common.CNOT
            elif gatetype == cirq_pauli._PauliX and gate.exponent == 1:
                gate = cirq_pauli.X
            elif gatetype == cirq_pauli._PauliY and gate.exponent == 1:
                gate = cirq_pauli.Y
            elif gatetype == cirq_pauli._PauliZ and gate.exponent == 1:
                gate = cirq_pauli.Z

            decomp_gate, decomp_param, decomp_qubit = None, None, None
            if gate in _constant_gates:
                try:
                    optype = _cirq2ops_mapping[gate]
                except KeyError as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
                params = []
            elif isinstance(gate, cirq_common.MeasurementGate):
                id = Bit(gate.key)
                tkcirc.add_bit(id)
                tkcirc.Measure(*qb_lst, id)
                continue
            elif isinstance(gate, cirq.PhasedXPowGate):
                optype = OpType.PhasedX
                pe = gate.phase_exponent
                e = gate.exponent
                params = [e, pe]
            elif isinstance(gate, cirq.CXPowGate) and gate.exponent != 1:
                optype = OpType.CRz
                params = [gate.exponent]
                decomp_gate = OpType.Ry
                decomp_param = [0.5]
                decomp_qubit = [qmap[op.qubits[-1]]]
                tkcirc.add_gate(decomp_gate, [-0.5], decomp_qubit)
            else:
                try:
                    optype = _cirq2ops_mapping[gatetype]
                except KeyError as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
                params = [gate.exponent]
            tkcirc.add_gate(optype, params, qb_lst)
            if decomp_gate is not None:
                tkcirc.add_gate(decomp_gate, decomp_param, decomp_qubit)
    return tkcirc
