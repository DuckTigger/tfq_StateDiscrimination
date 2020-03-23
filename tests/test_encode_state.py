import unittest
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy as sy

from encode_state import EncodeState
from input_circuits import InputCircuits


class TestEncodeState(unittest.TestCase):
    def test_encoding_layers(self):
        n = 6
        encoder = EncodeState(n)
        test_circuit = encoder.create_encoding_circuit()
        print(test_circuit)
        self.assertEqual(len(test_circuit), (3 * (n + 1)))

    def test_one_qubit_unitary(self):
        sym = sy.symbols('0:3')
        qubit = cirq.LineQubit.range(1)[0]
        unit = EncodeState.one_qubit_unitary(qubit, sym)
        circuit = cirq.Circuit(unit)
        thetas = [np.random.rand() for _ in range(3)]
        test = cirq.Circuit([cirq.X(qubit) ** thetas[0], cirq.Y(qubit) ** thetas[1], cirq.Z(qubit) ** thetas[2]])
        sim = cirq.Simulator()
        resolve = cirq.ParamResolver({'0': float(thetas[0]), '1': thetas[1], '2': thetas[2]})
        np.testing.assert_array_almost_equal(sim.simulate(circuit, param_resolver=resolve).final_state,
                                             test.final_wavefunction())

    def test_encode_state(self):
        n = 4
        encoder = EncodeState(n)
        symbols = sy.symbols('enc0:{}'.format(4 * n))
        circuit = encoder.create_encoding_circuit(symbols)
        readout = cirq.PauliString(1, cirq.Z(encoder.qubits[2]), cirq.Z(encoder.qubits[3]))
        enc = tfq.layers.PQC(circuit, readout)
        circuits = InputCircuits(n)
        data = tfq.convert_to_tensor([circuits.create_a(0.5), circuits.create_b(0.3, 0)])
        res = enc(data)
        print(res)
        return res

    def test_create_layers(self):
        n = 4
        encoder = EncodeState(n)
        circuit = cirq.Circuit()
        for i in range(n):
            symbols = sy.symbols('layer{}_0:{}'.format(i, 4 * n - i))
            layer = encoder.create_layers(symbols, i)
            circuit.append(layer)
        print(circuit.to_text_diagram(transpose=True))
        true = cirq.Circuit([cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer0_0')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer0_1')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer0_2')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer0_3')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer0_4')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer0_5')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer0_6')).on(cirq.GridQubit(1, 0)),
            (cirq.CNOT ** sy.Symbol('layer0_12')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer0_7')).on(cirq.GridQubit(1, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer0_8')).on(cirq.GridQubit(1, 0)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer0_9')).on(cirq.GridQubit(1, 1)),
            (cirq.CNOT ** sy.Symbol('layer0_13')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer0_10')).on(cirq.GridQubit(1, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer0_11')).on(cirq.GridQubit(1, 1)),
        ]), cirq.Moment(operations=[
            (cirq.CNOT ** sy.Symbol('layer0_14')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
        ]), cirq.Moment(operations=[
            (cirq.CNOT ** sy.Symbol('layer0_15')).on(cirq.GridQubit(1, 1), cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            cirq.MeasurementGate(1, 'm0', ()).on(cirq.GridQubit(1, 1)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer1_0')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer1_1')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer1_2')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer1_3')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer1_4')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer1_5')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer1_6')).on(cirq.GridQubit(1, 0)),
            (cirq.CNOT ** sy.Symbol('layer1_9')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer1_7')).on(cirq.GridQubit(1, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer1_8')).on(cirq.GridQubit(1, 0)),
        ]), cirq.Moment(operations=[
            (cirq.CNOT ** sy.Symbol('layer1_10')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
        ]), cirq.Moment(operations=[
            (cirq.CNOT ** sy.Symbol('layer1_11')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            cirq.MeasurementGate(1, 'm1', ()).on(cirq.GridQubit(1, 0)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer2_0')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer2_1')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer2_2')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer2_3')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer2_4')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer2_5')).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.CNOT ** sy.Symbol('layer2_6')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.CNOT ** sy.Symbol('layer2_7')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            cirq.MeasurementGate(1, 'm2', ()).on(cirq.GridQubit(0, 1)),
        ]), cirq.Moment(operations=[
            (cirq.X ** sy.Symbol('layer3_0')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Y ** sy.Symbol('layer3_1')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            (cirq.Z ** sy.Symbol('layer3_2')).on(cirq.GridQubit(0, 0)),
        ]), cirq.Moment(operations=[
            cirq.MeasurementGate(1, 'm3', ()).on(cirq.GridQubit(0, 0)),
        ])
        ])
        cirq.testing.assert_same_circuits(circuit, true)

    def test_discrimination_circuit(self):
        n = 4
        encoder = EncodeState(n)
        test = encoder.discrimination_circuit()
        true = cirq.Circuit([cirq.Moment(operations=[
                (cirq.X ** sy.Symbol('layer0_0')).on(cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sy.Symbol('layer0_1')).on(cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sy.Symbol('layer0_2')).on(cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sy.Symbol('layer0_3')).on(cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sy.Symbol('layer0_4')).on(cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sy.Symbol('layer0_5')).on(cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sy.Symbol('layer0_6')).on(cirq.GridQubit(1, 0)),
                (cirq.CNOT ** sy.Symbol('layer0_12')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sy.Symbol('layer0_7')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sy.Symbol('layer0_8')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sy.Symbol('layer0_9')).on(cirq.GridQubit(1, 1)),
                (cirq.CNOT ** sy.Symbol('layer0_13')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sy.Symbol('layer0_10')).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sy.Symbol('layer0_11')).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sy.Symbol('layer0_14')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sy.Symbol('layer0_15')).on(cirq.GridQubit(1, 1), cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm0', ()).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sy.Symbol('layer1_0')).on(cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sy.Symbol('layer1_1')).on(cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sy.Symbol('layer1_2')).on(cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sy.Symbol('layer1_3')).on(cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sy.Symbol('layer1_4')).on(cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sy.Symbol('layer1_5')).on(cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sy.Symbol('layer1_6')).on(cirq.GridQubit(1, 0)),
                (cirq.CNOT ** sy.Symbol('layer1_9')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sy.Symbol('layer1_7')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sy.Symbol('layer1_8')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sy.Symbol('layer1_10')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sy.Symbol('layer1_11')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm1', ()).on(cirq.GridQubit(1, 0)),
            ])
            ])
        print(test.to_text_diagram(transpose=True))
        cirq.testing.assert_same_circuits(test, true)


if __name__ == '__main__':
    unittest.main()
