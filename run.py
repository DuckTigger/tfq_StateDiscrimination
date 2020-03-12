import tensorflow as tf

from encode_state import EncodeState
from input_circuits import InputCircuits


def main():
    n = 4
    circuits = InputCircuits(n)
    a_circuits, b_circuits = circuits.create_discrimination_circuits()
    encoder = EncodeState(n)
    encoding_model = encoder.encode_state(a_circuits)
    encoding_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss=tf.losses.mse)
    return encoding_model


if __name__ == '__main__':
    main()
