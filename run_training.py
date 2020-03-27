from argparse import ArgumentParser

from train_model import TrainModel


def main():
    parser = ArgumentParser(description='Runs tfq State Discrimination with the arguments passed here')
    parser.add_argument('--n_qubits', type=int, nargs='?', default=4,
                        help='The number of qubits in the state discrimination circuit')
    parser.add_argument('--epochs', type=int, nargs='?', default=7,
                        help='The number of epochs to train for')
    parser.add_argument('--batch_size', type=int, nargs='?', default=20,
                        help='The size of batches in training.')
    parser.add_argument('--mu_a', type=float, nargs='?', default=0.5,
                        help='The mean value for the a state distribution.')
    parser.add_argument('--err_loss', type=float, nargs='?', default=0.5,
                        help='The weight to minimise errors in the cost function. err_inc is given by 1 - err_loss.')
    parser.add_argument('--noise_level', type=float, nargs='?', default=None,
                        help='The noise level to apply. Default is no noise, do not use 0 as this will be slower.')
    parser.add_argument('--restire_loc', type=str, nargs='?', default=None,
                        help='Location of a model to restore from. Default is to create a new model')

    args = parser.parse_args()
    trainer = TrainModel(**vars(args))
    trainer.train_model()


if __name__ == '__main__':
    main()
