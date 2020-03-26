import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt

from encode_state import EncodeState
from input_circuits import InputCircuits
from loss import DiscriminationLoss


def main():
    n = 4
    circuits = InputCircuits(n)
    train_circuits, train_labels, test_circuits, test_labels = circuits.create_discrimination_circuits(mu_a=0.9)
    encoder = EncodeState(n)
    pqc_model = encoder.encode_state_PQC()
    discrimination_model = encoder.discrimination_model()
    # controlled_model = encoder.discrimination_model(True)
    model = discrimination_model

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
