import tensorflow as tf
import matplotlib.pyplot as plt

from encode_state import EncodeState
from input_circuits import InputCircuits
from loss import DiscrimintaionLoss


def main():
    n = 4
    circuits = InputCircuits(n)
    train_circuits, train_labels, test_circuits, test_labels = circuits.create_discrimination_circuits()
    encoder = EncodeState(n)
    encoding_model = encoder.encode_state(train_circuits)

    loss = DiscrimintaionLoss(0.5, 0.5)
    loss_fn = loss.discrimination_loss
    encoding_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss=loss_fn)

    history = encoding_model.fit(x=train_circuits,
                                 y=train_labels,
                                 batch_size=10,
                                 epochs=2,
                                 verbose=1,
                                 validation_data=(test_circuits, test_labels))
    plt.plot(history.history['loss'], label='Training')
    plt.show()


if __name__ == '__main__':
    main()
