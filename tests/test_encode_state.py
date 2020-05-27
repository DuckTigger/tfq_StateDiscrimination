import unittest
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy as sp

from encode_state import EncodeState
from leakage import LeakageModels
from input_circuits import InputCircuits
from circuit_layers import CircuitLayers


class TestEncodeState(unittest.TestCase):
    def test_encoding_layers(self):
        n = 6
        encoder = EncodeState(n)
        qubits = cirq.GridQubit.rect(int(n/2), 2)
        test_circuit = CircuitLayers.create_encoding_circuit(qubits)
        true = cirq.Circuit(
            [cirq.Moment(operations=[
    cirq.H.on(cirq.GridQubit(0, 0)),
    cirq.H.on(cirq.GridQubit(0, 1)),
    cirq.H.on(cirq.GridQubit(1, 0)),
    cirq.H.on(cirq.GridQubit(1, 1)),
    cirq.H.on(cirq.GridQubit(2, 0)),
    cirq.H.on(cirq.GridQubit(2, 1)),
]), cirq.Moment(operations=[
    (cirq.X**sp.Symbol('enc0')).on(cirq.GridQubit(0, 0)),
    (cirq.X**sp.Symbol('enc3')).on(cirq.GridQubit(0, 1)),
    (cirq.X**sp.Symbol('enc6')).on(cirq.GridQubit(1, 0)),
    (cirq.X**sp.Symbol('enc9')).on(cirq.GridQubit(1, 1)),
    (cirq.X**sp.Symbol('enc12')).on(cirq.GridQubit(2, 0)),
    (cirq.X**sp.Symbol('enc15')).on(cirq.GridQubit(2, 1)),
]), cirq.Moment(operations=[
    (cirq.Y**sp.Symbol('enc1')).on(cirq.GridQubit(0, 0)),
    (cirq.Y**sp.Symbol('enc4')).on(cirq.GridQubit(0, 1)),
    (cirq.Y**sp.Symbol('enc7')).on(cirq.GridQubit(1, 0)),
    (cirq.Y**sp.Symbol('enc10')).on(cirq.GridQubit(1, 1)),
    (cirq.Y**sp.Symbol('enc13')).on(cirq.GridQubit(2, 0)),
    (cirq.Y**sp.Symbol('enc16')).on(cirq.GridQubit(2, 1)),
]), cirq.Moment(operations=[
    (cirq.Z**sp.Symbol('enc2')).on(cirq.GridQubit(0, 0)),
    (cirq.Z**sp.Symbol('enc5')).on(cirq.GridQubit(0, 1)),
    (cirq.Z**sp.Symbol('enc8')).on(cirq.GridQubit(1, 0)),
    (cirq.Z**sp.Symbol('enc11')).on(cirq.GridQubit(1, 1)),
    (cirq.Z**sp.Symbol('enc14')).on(cirq.GridQubit(2, 0)),
    (cirq.Z**sp.Symbol('enc17')).on(cirq.GridQubit(2, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('enc18')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('enc19')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('enc20')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('enc21')).on(cirq.GridQubit(1, 1), cirq.GridQubit(2, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('enc22')).on(cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)),
    cirq.CNOT.on(cirq.GridQubit(1, 1), cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('enc23')).on(cirq.GridQubit(2, 1), cirq.GridQubit(0, 0)),
    cirq.MeasurementGate(2, 'm', ()).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
])
            ])
        cirq.testing.assert_same_circuits(test_circuit, true)

    def test_one_qubit_unitary(self):
        sym = sp.symbols('0:3')
        qubit = cirq.LineQubit.range(1)[0]
        unit = CircuitLayers.one_qubit_unitary(qubit, sym)
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
        qubits = cirq.GridQubit.rect(int(n / 2), 2)
        symbols = sp.symbols('enc0:{}'.format(4 * n))
        circuit = CircuitLayers.create_encoding_circuit(qubits, symbols)
        readout = cirq.PauliString(1, cirq.Z(encoder.qubits[2]), cirq.Z(encoder.qubits[3]))
        enc = tfq.layers.PQC(circuit, readout)
        circuits = InputCircuits(n)
        data = tfq.convert_to_tensor([circuits.create_a(0.5), circuits.create_b(0.3, 0)])
        res = enc(data)
        print(res)
        return res

    def test_discrimination_model(self):
        n = 4
        encoder = EncodeState(n)
        circuits = InputCircuits(n)
        qubits = cirq.GridQubit.rect(int(n / 2), 2)
        model = encoder.discrimination_model()
        data = tfq.convert_to_tensor([circuits.create_a(0.5), circuits.create_b(0.3, 0)])
        res = model(data)
        print(res)
        return res

    def test_create_layers(self):
        n = 4
        encoder = EncodeState(n)
        circuit = cirq.Circuit()
        qubits = cirq.GridQubit.rect(int(n / 2), 2)
        for i in range(n):
            symbols = sp.symbols('layer{}_0:{}'.format(i, 4 * n - i))
            layer = CircuitLayers.create_layers(qubits, symbols, i)
            circuit.append(layer)
        print(circuit.to_text_diagram(transpose=True))
        true = cirq.Circuit(
            [cirq.Moment(operations=[
    (cirq.X**sp.Symbol('layer0_0')).on(cirq.GridQubit(0, 0)),
    (cirq.X**sp.Symbol('layer0_3')).on(cirq.GridQubit(0, 1)),
    (cirq.X**sp.Symbol('layer0_6')).on(cirq.GridQubit(1, 0)),
    (cirq.X**sp.Symbol('layer0_9')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.Y**sp.Symbol('layer0_1')).on(cirq.GridQubit(0, 0)),
    (cirq.Y**sp.Symbol('layer0_4')).on(cirq.GridQubit(0, 1)),
    (cirq.Y**sp.Symbol('layer0_7')).on(cirq.GridQubit(1, 0)),
    (cirq.Y**sp.Symbol('layer0_10')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.Z**sp.Symbol('layer0_2')).on(cirq.GridQubit(0, 0)),
    (cirq.Z**sp.Symbol('layer0_5')).on(cirq.GridQubit(0, 1)),
    (cirq.Z**sp.Symbol('layer0_8')).on(cirq.GridQubit(1, 0)),
    (cirq.Z**sp.Symbol('layer0_11')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_12')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_13')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_14')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_15')).on(cirq.GridQubit(1, 1), cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm0', ()).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.X**sp.Symbol('layer1_0')).on(cirq.GridQubit(0, 0)),
    (cirq.X**sp.Symbol('layer1_3')).on(cirq.GridQubit(0, 1)),
    (cirq.X**sp.Symbol('layer1_6')).on(cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.Y**sp.Symbol('layer1_1')).on(cirq.GridQubit(0, 0)),
    (cirq.Y**sp.Symbol('layer1_4')).on(cirq.GridQubit(0, 1)),
    (cirq.Y**sp.Symbol('layer1_7')).on(cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.Z**sp.Symbol('layer1_2')).on(cirq.GridQubit(0, 0)),
    (cirq.Z**sp.Symbol('layer1_5')).on(cirq.GridQubit(0, 1)),
    (cirq.Z**sp.Symbol('layer1_8')).on(cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer1_9')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer1_10')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer1_11')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm1', ()).on(cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.X**sp.Symbol('layer2_0')).on(cirq.GridQubit(0, 0)),
    (cirq.X**sp.Symbol('layer2_3')).on(cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.Y**sp.Symbol('layer2_1')).on(cirq.GridQubit(0, 0)),
    (cirq.Y**sp.Symbol('layer2_4')).on(cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.Z**sp.Symbol('layer2_2')).on(cirq.GridQubit(0, 0)),
    (cirq.Z**sp.Symbol('layer2_5')).on(cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer2_6')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer2_7')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm2', ()).on(cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.X**sp.Symbol('layer3_0')).on(cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    (cirq.Y**sp.Symbol('layer3_1')).on(cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    (cirq.Z**sp.Symbol('layer3_2')).on(cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm3', ()).on(cirq.GridQubit(0, 0)),
])
            ])
        cirq.testing.assert_same_circuits(circuit, true)

    def test_discrimination_circuit(self):
        n = 4
        encoder = EncodeState(n)
        test = encoder.discrimination_circuit()
        true = cirq.Circuit([
            [cirq.Moment(operations=[
                (cirq.X ** sp.Symbol('layer0_0')).on(cirq.GridQubit(0, 0)),
                (cirq.X ** sp.Symbol('layer0_3')).on(cirq.GridQubit(0, 1)),
                (cirq.X ** sp.Symbol('layer0_6')).on(cirq.GridQubit(1, 0)),
                (cirq.X ** sp.Symbol('layer0_9')).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer0_1')).on(cirq.GridQubit(0, 0)),
                (cirq.Y ** sp.Symbol('layer0_4')).on(cirq.GridQubit(0, 1)),
                (cirq.Y ** sp.Symbol('layer0_7')).on(cirq.GridQubit(1, 0)),
                (cirq.Y ** sp.Symbol('layer0_10')).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sp.Symbol('layer0_2')).on(cirq.GridQubit(0, 0)),
                (cirq.Z ** sp.Symbol('layer0_5')).on(cirq.GridQubit(0, 1)),
                (cirq.Z ** sp.Symbol('layer0_8')).on(cirq.GridQubit(1, 0)),
                (cirq.Z ** sp.Symbol('layer0_11')).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_12')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_13')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_14')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer0_15')).on(cirq.GridQubit(1, 1), cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm0', ()).on(cirq.GridQubit(1, 1)),
            ]), cirq.Moment(operations=[
                (cirq.X ** sp.Symbol('layer1_0')).on(cirq.GridQubit(0, 0)),
                (cirq.X ** sp.Symbol('layer1_3')).on(cirq.GridQubit(0, 1)),
                (cirq.X ** sp.Symbol('layer1_6')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Y ** sp.Symbol('layer1_1')).on(cirq.GridQubit(0, 0)),
                (cirq.Y ** sp.Symbol('layer1_4')).on(cirq.GridQubit(0, 1)),
                (cirq.Y ** sp.Symbol('layer1_7')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.Z ** sp.Symbol('layer1_2')).on(cirq.GridQubit(0, 0)),
                (cirq.Z ** sp.Symbol('layer1_5')).on(cirq.GridQubit(0, 1)),
                (cirq.Z ** sp.Symbol('layer1_8')).on(cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer1_9')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer1_10')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
            ]), cirq.Moment(operations=[
                (cirq.CNOT ** sp.Symbol('layer1_11')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
            ]), cirq.Moment(operations=[
                cirq.MeasurementGate(1, 'm1', ()).on(cirq.GridQubit(1, 0)),
            ])]
        ])
        print(test.to_text_diagram(transpose=True))
        cirq.testing.assert_same_circuits(test, true)

    def test_create_controlled_layers(self):
        n = 4
        encoder = EncodeState(n)
        circuit = cirq.Circuit()
        qubits = cirq.GridQubit.rect(int(n / 2), 2)
        for i in range(n):
            symbols = sp.symbols('layer{}_0:{}'.format(1, 4 * n - i))
            layer = CircuitLayers.create_layers(qubits, symbols, i, True)
            circuit.append(layer)
        true = cirq.Circuit(
            [[cirq.Moment(operations=[
    (cirq.X**sp.Symbol('layer1_0')).on(cirq.GridQubit(0, 0)),
    (cirq.X**sp.Symbol('layer1_3')).on(cirq.GridQubit(0, 1)),
    (cirq.X**sp.Symbol('layer1_6')).on(cirq.GridQubit(1, 0)),
    (cirq.X**sp.Symbol('layer1_9')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.Y**sp.Symbol('layer1_1')).on(cirq.GridQubit(0, 0)),
    (cirq.Y**sp.Symbol('layer1_4')).on(cirq.GridQubit(0, 1)),
    (cirq.Y**sp.Symbol('layer1_7')).on(cirq.GridQubit(1, 0)),
    (cirq.Y**sp.Symbol('layer1_10')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.Z**sp.Symbol('layer1_2')).on(cirq.GridQubit(0, 0)),
    (cirq.Z**sp.Symbol('layer1_5')).on(cirq.GridQubit(0, 1)),
    (cirq.Z**sp.Symbol('layer1_8')).on(cirq.GridQubit(1, 0)),
    (cirq.Z**sp.Symbol('layer1_11')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer1_12')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer1_13')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer1_14')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer1_15')).on(cirq.GridQubit(1, 1), cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm0', ()).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_0_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_3_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_4_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_5_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_6_0')).on(cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_7_0')).on(cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_8_0')).on(cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_9_0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_10_0')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_11_0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_0_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_3_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_4_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_5_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_6_1')).on(cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_7_1')).on(cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_8_1')).on(cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_9_1')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_10_1')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_11_1')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm1', ()).on(cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.X**sp.Symbol('layer1_0_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.X**sp.Symbol('layer1_3_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Y**sp.Symbol('layer1_4_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Z**sp.Symbol('layer1_5_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_6_0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_7_0')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.X**sp.Symbol('layer1_0_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.X**sp.Symbol('layer1_3_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Y**sp.Symbol('layer1_4_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.Z**sp.Symbol('layer1_5_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_6_1')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 0),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_7_1')).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm2', ()).on(cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(0, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_0_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(0, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(0, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(0, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_0_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(0, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(0, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm3', ()).on(cirq.GridQubit(0, 0)),
])]
        ])
        print(circuit.to_text_diagram(transpose=True))
        cirq.testing.assert_same_circuits(circuit, true)

    def test_controlled_discrimination_circuit(self):
        n = 4
        encoder = EncodeState(n)
        test = encoder.discrimination_circuit(True)
        print(test.to_text_diagram(transpose=True))
        true = cirq.Circuit(
            [cirq.Moment(operations=[
    (cirq.X**sp.Symbol('layer0_0')).on(cirq.GridQubit(0, 0)),
    (cirq.X**sp.Symbol('layer0_3')).on(cirq.GridQubit(0, 1)),
    (cirq.X**sp.Symbol('layer0_6')).on(cirq.GridQubit(1, 0)),
    (cirq.X**sp.Symbol('layer0_9')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.Y**sp.Symbol('layer0_1')).on(cirq.GridQubit(0, 0)),
    (cirq.Y**sp.Symbol('layer0_4')).on(cirq.GridQubit(0, 1)),
    (cirq.Y**sp.Symbol('layer0_7')).on(cirq.GridQubit(1, 0)),
    (cirq.Y**sp.Symbol('layer0_10')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.Z**sp.Symbol('layer0_2')).on(cirq.GridQubit(0, 0)),
    (cirq.Z**sp.Symbol('layer0_5')).on(cirq.GridQubit(0, 1)),
    (cirq.Z**sp.Symbol('layer0_8')).on(cirq.GridQubit(1, 0)),
    (cirq.Z**sp.Symbol('layer0_11')).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_12')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_13')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_14')).on(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    (cirq.CNOT**sp.Symbol('layer0_15')).on(cirq.GridQubit(1, 1), cirq.GridQubit(0, 0)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm0', ()).on(cirq.GridQubit(1, 1)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_0_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_0')).on(cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_3_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_4_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_5_0')).on(cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_6_0')).on(cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_7_0')).on(cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_8_0')).on(cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_9_0')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_10_0')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_11_0')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)), control_values=((0,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_0_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_1_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_2_1')).on(cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_3_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_4_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_5_1')).on(cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.X**sp.Symbol('layer1_6_1')).on(cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Y**sp.Symbol('layer1_7_1')).on(cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.Z**sp.Symbol('layer1_8_1')).on(cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_9_1')).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_10_1')).on(cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.ControlledOperation(controls=(cirq.GridQubit(1, 1),), sub_operation=(cirq.CNOT**sp.Symbol('layer1_11_1')).on(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)), control_values=((1,),)),
]), cirq.Moment(operations=[
    cirq.MeasurementGate(1, 'm1', ()).on(cirq.GridQubit(1, 0)),
])
            ])
        cirq.testing.assert_same_circuits(test, true)

    def test_leakage_circuits(self):
        n = 4
        leakage = LeakageModels(int(n/2), int(n/2), False, 0.1)
        test = leakage.leaky_discrimination_circuit()
        print(test.to_text_diagram(transpose=True))

    def test_leakage_circuits_data_train(self):
        n = 4
        leakage = LeakageModels(int(n/2), int(n/2), True, 0.1)
        test = leakage.leaky_discrimination_circuit()
        print(test.to_text_diagram(transpose=True))


if __name__ == '__main__':
    unittest.main()
