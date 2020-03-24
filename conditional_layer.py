import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import sympy as sp
from typing import List, Union, Dict


class ConditionalQuantumLayer(tf.keras.layers.Layer):
    tfq.layers.PQC