import numbers
import tensorflow as tf
import cirq
from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_executors import expectation, sampled_expectation
from tensorflow_quantum.python.layers.circuit_construction import elementary
import numpy as np
import sympy
from typing import List, Union, Dict


class ConditionalQuantumLayer(tf.keras.layers.Layer):
    """
    Calls two different PQCs dependent upon the outcome passed to it - this allows for classical control
    """
    def __init__(self,
                 model_circuit,
                 operators,
                 *,
                 repetitions=None,
                 backend=None,
                 differentiator=None,
                 initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
                 regularizer=None,
                 constraint=None,
                 **kwargs,):
        super().__init__(**kwargs)

        # Ingest model_circuit.
        if not isinstance(model_circuit, cirq.Circuit):
            raise TypeError("model_circuit must be a cirq.Circuit object."
                            " Given: {}".format(model_circuit))

        self._symbols_list = list(
            sorted(util.get_circuit_symbols(model_circuit)))
        self._symbols = tf.constant([str(x) for x in self._symbols_list])

        self._model_circuit = util.convert_to_tensor([model_circuit])
        if len(self._symbols_list) == 0:
            raise ValueError("model_circuit has no sympy.Symbols. Please "
                             "provide a circuit that contains symbols so "
                             "that their values can be trained.")

        # Ingest operators.
        if isinstance(operators, (cirq.PauliString, cirq.PauliSum)):
            operators = [operators]
        if not isinstance(operators, (list, np.ndarray, tuple)):
            raise TypeError("operators must be a cirq.PauliSum or "
                            "cirq.PauliString, or a list, tuple, "
                            "or np.array containing them. "
                            "Got {}.".format(type(operators)))
        if not all([
            isinstance(op, (cirq.PauliString, cirq.PauliSum))
            for op in operators
        ]):
            raise TypeError("Each element in operators to measure "
                            "must be a cirq.PauliString"
                            " or cirq.PauliSum")
        self._operators = util.convert_to_tensor([operators])

        # Ingest and promote repetitions.
        self._analytic = False
        if repetitions is None:
            self._analytic = True
        if not self._analytic and not isinstance(repetitions, numbers.Integral):
            raise TypeError("repetitions must be a positive integer value."
                            " Given: ".format(repetitions))
        if not self._analytic and repetitions <= 0:
            raise ValueError("Repetitions must be greater than zero.")
        if not self._analytic:
            self._repetitions = tf.constant(
                [[repetitions for _ in range(len(operators))]],
                dtype=tf.dtypes.int32)

        # Set backend and differentiator.
        if not isinstance(backend, cirq.Sampler
                          ) and repetitions is not None and backend is not None:
            raise TypeError("provided backend does not inherit cirq.Sampler "
                            "and repetitions!=None. Please provide a backend "
                            "that inherits cirq.Sampler or set "
                            "repetitions=None.")
        if not isinstance(backend, cirq.SimulatesFinalState
                          ) and repetitions is None and backend is not None:
            raise TypeError("provided backend does not inherit "
                            "cirq.SimulatesFinalState and repetitions=None. "
                            "Please provide a backend that inherits "
                            "cirq.SimulatesFinalState or choose a positive "
                            "number of repetitions.")
        if self._analytic:
            self._executor = expectation.Expectation(
                backend=backend, differentiator=differentiator)
        else:
            self._executor = sampled_expectation.SampledExpectation(
                backend=backend, differentiator=differentiator)

        self._append_layer = elementary.AddCircuit()

        # Set additional parameter controls.
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        # Weight creation is not placed in a Build function because the number
        # of weights is independent of the input shape.
        self.parameters = self.add_weight('parameters',
                                          shape=self._symbols.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          dtype=tf.float32,
                                          trainable=True)

    @property
    def symbols(self):
        """The symbols that are managed by this layer (in-order).

        Note: `symbols[i]` indicates what symbol name the managed variables in
            this layer map to.
        """
        return [sympy.Symbol(x) for x in self._symbols_list]

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs):
        """Keras call function."""
        circuit_batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_model = tf.tile(self._model_circuit, [circuit_batch_dim])
        model_appended = self._append_layer(inputs[0], append=tiled_up_model)
        tiled_up_parameters = tf.tile([self.parameters], [circuit_batch_dim, 1])
        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])

        # this is disabled to make autograph compilation easier.
        # pylint: disable=no-else-return
        symbols_1 = sympy.symbols(tuple([x.name + '_1' for x in self._symbols]))
        symbols_0 = sympy.symbols(tuple([x.name + '_0' for x in self._symbols]))
        measurement = tf.divide(tf.add(inputs[1], 1), 2)
        if self._analytic:
            expectation_0 = self._executor(model_appended,
                                            symbol_names=symbols_0,
                                            symbol_values=tiled_up_parameters,
                                            operators=tiled_up_operators)
            expectation_1 = self._executor(model_appended,
                                            symbol_names=symbols_1,
                                            symbol_values=tiled_up_parameters,
                                            operators=tiled_up_operators)
            return tf.add(tf.multiply(expectation_0, tf.subtract(1, measurement)),
                          tf.multiply(expectation_1, measurement))
        else:
            tiled_up_repetitions = tf.tile(self._repetitions,
                                           [circuit_batch_dim, 1])
            expectation_0 = self._executor(model_appended,
                                            symbol_names=symbols_0,
                                            symbol_values=tiled_up_parameters,
                                            operators=tiled_up_operators,
                                            repetitions=tiled_up_repetitions)
            expectation_1 = self._executor(model_appended,
                                            symbol_names=symbols_1,
                                            symbol_values=tiled_up_parameters,
                                            operators=tiled_up_operators,
                                            repetitions=tiled_up_repetitions)
            return tf.add(tf.multiply(expectation_0, tf.subtract(1, measurement)),
                          tf.multiply(expectation_1, measurement))
        # pylint: enable=no-else-return
