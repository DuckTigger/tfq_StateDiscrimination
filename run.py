import tensorflow as tf
import cirq
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt

from encode_state import EncodeState
from leakage import LeakageModels
from qutrit_model import QutritModel
from input_circuits import InputCircuits
from loss import DiscriminationLoss
from noise.noise_model import TwoQubitNoiseModel, two_qubit_depolarize


def main():
    n = 4
    circuits = InputCircuits(n)
    train_circuits, train_labels, test_circuits, test_labels = circuits.create_discrimination_circuits(mu_a=0.9)
    encoder = EncodeState(n)
    leakage = LeakageModels(2, 2, False, 0.3)
    qutrits = QutritModel(2, 0.1)
    noise_model = TwoQubitNoiseModel(cirq.depolarize(0.01), two_qubit_depolarize(0.01))
    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    # pqc_model = encoder.encode_state_PQC()
    # discrimination_model = encoder.discrimination_model()
    # controlled_model = encoder.discrimination_model(True)
    # noisy_discrimination = encoder.discrimination_model(backend=noisy_sim)
    # leakage_model = leakage.leaky_model()
    qutrit_model = qutrits.qutrit_model()
    model = qutrit_model

    loss = DiscriminationLoss(0.5, 0.5)
    loss_fn = loss.discrimination_loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=loss_fn)

    history = model.fit(x=train_circuits,
                        y=train_labels,
                        batch_size=10,
                        epochs=7,
                        verbose=1,
                        validation_data=(test_circuits, test_labels))
    plt.plot(history.history['loss'], label='Training')
    plt.show()


if __name__ == '__main__':
    main()
