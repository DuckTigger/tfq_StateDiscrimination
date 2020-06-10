import tensorflow as tf


class DiscriminationLoss:

    def __init__(self, w_error: float, w_inconclusive: float):
        self.w_inconclusive = w_inconclusive
        self.w_error = w_error

    @tf.function
    def discrimination_loss(self, y_label, y_measurement):
        y_label = tf.cast(y_label, tf.float32)
        error, inconclusive = tf.map_fn(lambda x: self.measurement_to_loss(x[0], x[1]), (y_measurement, y_label))
        loss_vec = tf.add(error, inconclusive)
        return tf.reduce_mean(loss_vec)

    def measurement_to_loss(self, measurement: tf.Tensor, label: tf.Tensor):
        measurement = tf.squeeze(measurement)
        probs = self.m_outcome_to_probs(measurement)
        return self.probs_to_err_inc(label, probs)

    def probs_to_err_inc(self, label: tf.Tensor, probs: tf.Tensor):
        # 1 == a, 0 == b
        fn_a = lambda: tf.gather(probs, 1)
        fn_b = lambda: tf.add(tf.gather(probs, 0), tf.gather(probs, 2))
        error = tf.case([(tf.equal(label, 1), fn_a)], default=fn_b)
        error *= self.w_error
        inconclusive = tf.gather(probs, 3)
        inconclusive *= self.w_inconclusive
        return error, inconclusive

    @staticmethod
    def m_outcome_to_probs(measurement: tf.Tensor):
        measurement = tf.divide(tf.add(measurement, 1), 2)
        prob_0 = lambda x: 1 - x
        qubit_0 = tf.gather(measurement, 0)
        qubit_1 = tf.gather(measurement, 1)
        prob_00 = tf.add(prob_0(qubit_0), prob_0(qubit_1)) / 4
        prob_01 = tf.add(prob_0(qubit_0), qubit_1) / 4
        prob_10 = tf.add(qubit_0, prob_0(qubit_1)) / 4
        prob_11 = tf.reduce_sum(measurement) / 4
        return tf.stack([prob_00, prob_01, prob_10, prob_11])