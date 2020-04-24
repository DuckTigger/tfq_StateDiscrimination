import unittest
import numpy as np
import sympy as sp
import cirq
import tensorflow_quantum as tfq

from model_circuits import ModelCircuits
from leakage import LeakageModels


class TestLeakageCircuits(unittest.TestCase):
    def test_print_circuit(self):
        n_work = 2
        qubits = cirq.LineQubit.range(3 * n_work)
        work = qubits[:n_work]
        ancilla = qubits[n_work:2*n_work]
        readout = qubits[2*n_work:3*n_work]

        circuit = cirq.Circuit()
        for level in range(len(work)):
            symbols = sp.symbols('layer{}_0:{}'.format(level, 4 * n_work - level))
            circuit += ModelCircuits.create_leakage_layers(work, ancilla, readout, symbols, level)
        hard_coded = cirq.Circuit([
                cirq.Moment(operations=[
                (cirq.X ** sp.Symbol('layer0_0')).on(cirq.LineQubit(0)),
                (cirq.X ** sp.Symbol('layer0_3')).on(cirq.LineQubit(1)),
                (cirq.X ** 0.5).on(cirq.LineQubit(5)),
                (cirq.X ** 0.5).on(cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer0_1')).on(cirq.LineQubit(0)),
                (cirq.Y ** sp.Symbol('layer0_4')).on(cirq.LineQubit(1)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sp.Symbol('layer0_2')).on(cirq.LineQubit(0)),
                (cirq.Z ** sp.Symbol('layer0_5')).on(cirq.LineQubit(1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_6')).on(cirq.LineQubit(0), cirq.LineQubit(1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_7')).on(cirq.LineQubit(1), cirq.LineQubit(0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(0)),
                cirq.CNOT.on(cirq.LineQubit(3), cirq.LineQubit(1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer0_0_leak')).on(cirq.LineQubit(2)),
                (cirq.Y ** sp.Symbol('layer0_1_leak')).on(cirq.LineQubit(3)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(2)),
                cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(3)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Mul(sp.Integer(-1), sp.Symbol('layer0_0_leak'))).on(cirq.LineQubit(2)),
                (cirq.Y ** sp.Mul(sp.Integer(-1), sp.Symbol('layer0_1_leak'))).on(cirq.LineQubit(3)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(2)),
                cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(3)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(0)),
                cirq.CNOT.on(cirq.LineQubit(3), cirq.LineQubit(1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                (cirq.X ** 0.5).on(cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(3), cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(3), cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm0', ()).on(cirq.LineQubit(5)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sp.Symbol('layer1_0')).on(cirq.LineQubit(0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer1_1')).on(cirq.LineQubit(0)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sp.Symbol('layer1_2')).on(cirq.LineQubit(0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer1_0_leak')).on(cirq.LineQubit(2)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(2)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Mul(sp.Integer(-1), sp.Symbol('layer1_0_leak'))).on(cirq.LineQubit(2)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(2)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                (cirq.X ** 0.5).on(cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(4)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm1', ()).on(cirq.LineQubit(4)),
            ])
            ])
        cirq.testing.assert_same_circuits(circuit, hard_coded)

    def test_qubits_distinct(self):
        for n in range(1, 6):
            model = LeakageModels(n, n, False)
            data_set = set(model.data_qubits)
            ancilla_set = set(model.ancilla_qubits)
            readout_set = set(model.readout_qubits)
            self.assertEqual(len(data_set.difference(ancilla_set)), len(data_set))
            self.assertEqual(len(data_set.difference(readout_set)), len(data_set))
            self.assertEqual(len(ancilla_set.difference(readout_set)), len(ancilla_set))

    def test_leakage_model_no_data_train(self):
        model = LeakageModels(2, 2, False)
        circuit = model.leaky_discrimination_circuit()
        hard_coded = cirq.Circuit([
            cirq.Moment(operations=[
                (cirq.X ** sp.Symbol('layer0_0')).on(cirq.GridQubit(1, 0)),
                (cirq.X ** sp.Symbol('layer0_3')).on(cirq.GridQubit(1, 1)),
                (cirq.X ** 0.5).on(cirq.GridQubit(3, 1)),
                (cirq.X ** 0.5).on(cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer0_1')).on(cirq.GridQubit(1, 0)),
                (cirq.Y ** sp.Symbol('layer0_4')).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sp.Symbol('layer0_2')).on(cirq.GridQubit(1, 0)),
                (cirq.Z ** sp.Symbol('layer0_5')).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_6')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_7')).on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 0), cirq.GridQubit(1, 0)),
                cirq.CNOT.on(cirq.GridQubit(2, 1), cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer0_0_leak')).on(cirq.GridQubit(2, 0)),
                (cirq.Y ** sp.Symbol('layer0_1_leak')).on(cirq.GridQubit(2, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                cirq.CNOT.on(cirq.GridQubit(1, 1), cirq.GridQubit(2, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Mul(sp.Integer(-1), sp.Symbol('layer0_0_leak'))).on(cirq.GridQubit(2, 0)),
                (cirq.Y ** sp.Mul(sp.Integer(-1), sp.Symbol('layer0_1_leak'))).on(cirq.GridQubit(2, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                cirq.CNOT.on(cirq.GridQubit(1, 1), cirq.GridQubit(2, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 0), cirq.GridQubit(1, 0)),
                cirq.CNOT.on(cirq.GridQubit(2, 1), cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 1), cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** 0.5).on(cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 1), cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 1), cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 1), cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 1), cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm0', ()).on(cirq.GridQubit(3, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sp.Symbol('layer1_0')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer1_1')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sp.Symbol('layer1_2')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 0), cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer1_0_leak')).on(cirq.GridQubit(2, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Mul(sp.Integer(-1), sp.Symbol('layer1_0_leak'))).on(cirq.GridQubit(2, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 0), cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                (cirq.X ** 0.5).on(cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                (cirq.X ** -0.5).on(cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                cirq.CNOT.on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm1', ()).on(cirq.GridQubit(3, 0)),
            ])
            ])
        cirq.testing.assert_same_circuits(circuit, hard_coded)


if __name__ == '__main__':
    unittest.main()
