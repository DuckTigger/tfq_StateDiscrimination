import tensorflow as tf
import cirq
import matplotlib.pyplot as plt
import datetime
import copy
import sys
import os
import json
from typing import List, Tuple, Dict

from encode_state import EncodeState
from input_circuits import InputCircuits
from loss import DiscriminationLoss
from noise.noise_model import TwoQubitNoiseModel, two_qubit_depolarize


class TrainModel:

    def __init__(self, epochs: int = 7, batch_size: int = 20,
                 n_qubits: int = 4, mu_a: float = 0.5, err_loss: float = 0.5, learning_rate: float = 0.01,
                 restore_loc: str = None, noise_level: float = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n = n_qubits
        self.mu_a = mu_a
        self.err_loss = err_loss
        self.inc_loss = 1 - err_loss
        self.lr = learning_rate
        self.noise_level = noise_level
        params = copy.copy(locals())
        params.pop('self')
        self.save_loc = self.set_save_loc() if restore_loc is None else restore_loc
        self.callback, self.writer = self.setup_save(self.save_loc, params)

    @staticmethod
    def set_save_loc():
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if sys.platform.startswith('win'):
            save_loc = os.path.join(
                "C:\\Users\\Andrew Patterson\\Documents\\PhD\\tfq_state_discrimination\\checkpoints", time)
        else:
            if 'zcapga1' in os.getcwd():
                save_loc = os.path.join("/home/zcapga1/Scratch/tfq_state_discrimination/training_out", time)
            else:
                save_loc = os.path.join("/home/andrew/Documents/PhD/tfq_StateDiscrimination/training_out")
        return save_loc

    @staticmethod
    def setup_save(save_dir: str, params: Dict) -> Tuple[tf.keras.callbacks.ModelCheckpoint, tf.summary.SummaryWriter]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
            json.dump(params, f)
        ckpt_path = os.path.join(save_dir, 'cp-{epoch:04d}.ckpt')
        callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, verbose=1, save_weights_only=True, save_freq=2)
        writer = tf.summary.create_file_writer(save_dir)
        return callback, writer

    def train_model(self):
        circuits = InputCircuits(self.n)
        train_circuits, train_labels, test_circuits, test_labels = circuits.create_discrimination_circuits(
            mu_a=self.mu_a)
        encoder = EncodeState(self.n)

        if self.noise_level is not None:
            noise_model = TwoQubitNoiseModel(cirq.depolarize(4 * self.noise_level / 5),
                                             two_qubit_depolarize(self.noise_level))
            backend = cirq.DensityMatrixSimulator(noise=noise_model)
        else:
            backend = None
        model = encoder.discrimination_model(backend=backend)
        loss = DiscriminationLoss(self.err_loss, self.inc_loss)
        loss_fn = loss.discrimination_loss
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=loss_fn)
        history = model.fit(x=train_circuits,
                            y=train_labels,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(test_circuits, test_labels),
                            callbacks=[self.callback])
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training')
        ax.set_xlabel('step')
        ax.set_ylabel('loss')
        fig.savefig(os.path.join(self.save_loc, 'training_plot.png'))
        # model.save(os.path.join(self.save_loc, 'final_model.h5'))


if __name__ == '__main__':
    trainer = TrainModel(noise_level=0.01)
    trainer.train_model()
