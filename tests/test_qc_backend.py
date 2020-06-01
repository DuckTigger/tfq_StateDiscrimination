import unittest
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy as sp
import numpy as np

from backends.qc_backend import QCBackend
from encode_state import EncodeState
from input_circuits import InputCircuits
from loss import DiscriminationLoss


class TestAerBackend(unittest.TestCase):

    def test_discrimination_circuits(self):
        backend = QCBackend()
        circuits = InputCircuits(4)
        encoder = EncodeState(4)
        train_circuits, train_labels, test_circuits, test_labels = circuits.create_discrimination_circuits(5, mu_a=0.9)

        model = encoder.discrimination_model(backend=backend, repetitions=10)
        loss = DiscriminationLoss(0.5, 0.5)
        loss_fn = loss.discrimination_loss
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
        model_history = model.fit(x=train_circuits, y=train_labels,
                                  batch_size=5, epochs=2, verbose=1,
                                  validation_data=(test_circuits, test_labels))

    def test_simple_circuit(self):
        q = cirq.GridQubit.rect(2, 1)
        backend = QCBackend()
        sym = sp.symbols('a')
        circuit = cirq.Circuit([cirq.X(q[0])** sym, cirq.CNOT(q[0], q[1])])
        readout = [cirq.Z(q[1])]
        tf_in = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        pqc = tfq.layers.PQC(circuit, readout, backend=backend, repetitions=10)(tf_in)
        model = tf.keras.Model(inputs=[tf_in], outputs=[pqc])
        input_circuits = tfq.convert_to_tensor([cirq.Circuit([cirq.Z(q[1])]), cirq.Circuit([cirq.X(q[1])])])
        expected = np.array([[-1], [1]], dtype=np.float32)
        loss = tf.keras.losses.MeanSquaredError()
        model([input_circuits]).numpy()

        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss)
        #
        # history = model.fit(x=input_circuits,
        #           y=expected,
        #           batch_size=2,
        #           epochs=2,
        #           verbose=1)



if __name__ == '__main__':
    unittest.main()
