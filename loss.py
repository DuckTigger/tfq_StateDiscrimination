import tensorflow as tf
from typing import List


class DiscrimintaionLoss:

    def __init__(self, w_error:float, w_inconclusive: float):
        self.w_inconclusive = w_inconclusive
        self.w_error = w_error

    #@tf.function
    def discrimination_loss(self, y_label, y_measurement):
        y_label = tf.cast(y_label, tf.float32)
        (error, inconclusive) = tf.map_fn(lambda x: self.measurement_to_loss(x[0], x[1]), (y_label, y_measurement))
        loss_vec = tf.add(error, inconclusive)
        return tf.reduce_mean(loss_vec)

    def measurement_to_loss(self, label: int, measurement: List):
        # 1 == a, 0 == b
        fn_a = lambda: tf.cast(tf.equal(measurement, [0, 1]), dtype=tf.float32)
        fn_b = lambda: tf.cast(tf.logical_or(tf.equal(measurement, [0, 0]), tf.equal(measurement, [1, 0])), dtype=tf.float32)
        error = tf.case([(tf.equal(label, 1), fn_a)], default=fn_b)
        error *= self.w_error
        inconclusive = tf.cast(tf.equal(measurement, [1, 1]), dtype=tf.float32)
        inconclusive *= self.w_inconclusive
        return (error, inconclusive)