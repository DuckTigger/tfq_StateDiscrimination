import unittest
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import sympy as sy
import cirq

from loss import DiscriminationLoss
from .test_encode_state import TestEncodeState


class TestDiscrimiationLoss(unittest.TestCase):
    def test_m_to_probs(self):
        measurement = tf.convert_to_tensor([np.random.rand() * 2 - 1 for _ in range(2)])
        probs = DiscriminationLoss.m_outcome_to_probs(measurement)
        self.assertEqual(tf.reduce_sum(probs), 1)

    def test_dummy_outcome(self):
        encode_test = TestEncodeState()
        for _ in range(10):
            measurements = encode_test.test_encode_state()
            measurements = tf.squeeze(measurements)
            probs = DiscriminationLoss.m_outcome_to_probs(measurements)
            self.assertAlmostEqual(1, tf.reduce_sum(probs).numpy(), 6)

    def test_probs_to_err_inc(self):
        # 1 == a, 0 == b
        probs_err_a = tf.convert_to_tensor([0, 1.0, 0, 0])
        probs_err_b = tf.convert_to_tensor([0.5, 0, 0.5, 0])
        probs_inc = tf.convert_to_tensor([0,0,0,1.0])
        loss_calc = DiscriminationLoss(1, 1)
        a = tf.constant(1)
        b = tf.constant(0)
        self.assertEqual(1, loss_calc.probs_to_err_inc(a, probs_err_a)[0])
        self.assertEqual(0, loss_calc.probs_to_err_inc(b, probs_err_a)[0])
        self.assertEqual(1, loss_calc.probs_to_err_inc(b, probs_err_b)[0])
        self.assertEqual(0, loss_calc.probs_to_err_inc(a, probs_err_b)[0])
        self.assertEqual(1, loss_calc.probs_to_err_inc(a, probs_inc)[1])
        self.assertEqual(1, loss_calc.probs_to_err_inc(b, probs_inc)[1])

