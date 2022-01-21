import cirq
import copy
import os
import tensorflow as tf
import datetime
import tensorflow_probability as tfp
import tensorflow_datasets as tfd
import numpy as np
from typing import List, Dict, Tuple, Union
import timeit

class RBM(tf.keras.Model):

    def __init__(self, qubits: List[cirq.Qid], target_device: cirq.Device = cirq.google.Bristlecone, n_visible: int = None,
                 n_hidden: int = None, batch_size: int = None, W: tf.Tensor = None, h_bias: tf.Tensor = None, v_bias: tf.Tensor = None,
                 data_in: tf.Tensor = None, hidden_density: int = None):
        super(RBM, self). __init__(qubits, target_device)
        self.allowed_ops = self.return_allowed_operations()
        if n_hidden is None:
            if hidden_density is None:
                n_hidden = len(qubits)
            else:
                n_hidden = len(qubits) * hidden_density
        if n_visible is None:
            n_visible = len(qubits)
        if batch_size is None:
            batch_size = 1

        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.rng = tf.random.set_seed(1234)

        if W is None:
            init_W = tf.random.uniform((n_visible, n_hidden),
                                           minval=- 4 * np.sqrt(6 / (n_hidden + n_visible)),
                                       maxval=4 * np.sqrt(6 / (n_hidden + n_visible)))
            W = tf.Variable(initial_value=init_W, dtype=tf.float32, name='W')

        if h_bias is None:
            h_bias = tf.Variable(initial_value=np.zeros(n_hidden), dtype=tf.float32, name='h_bias')
        if v_bias is None:
            v_bias = tf.Variable(initial_value=np.zeros(n_visible), dtype=tf.float32, name='v_bias')

        self.batch_size = batch_size
        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias
        self.params = [self.W, self.h_bias, self.v_bias]
        self.data_in = data_in

    @property
    def data_in(self):
        return self.__data_in

    @data_in.setter
    def data_in(self, data):
        self.__data_in = iter(data)

    @data_in.getter
    def data_in(self):
        return self.__data_in

    def free_energy(self, v_sample: tf.Tensor) -> tf.Tensor:
        visible_term = tf.tensordot(v_sample, self.v_bias, 1)
        exponent = tf.tensordot(v_sample, self.W, 1) + self.h_bias
        hidden_term = tf.reduce_sum(tf.math.log(1 + tf.math.exp(exponent)), axis=0)
        return tf.subtract(tf.multiply(hidden_term, -1), visible_term)

    def propagate_up(self, visible: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pre_activation = tf.tensordot(visible, self.W, 1) + self.h_bias
        post_activation = tf.nn.sigmoid(pre_activation, name='propagate_up')
        return pre_activation, post_activation

    def propagate_down(self, hidden: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pre_activation = tf.tensordot(hidden, tf.transpose(self.W), 1) + self.v_bias
        post_activation = tf.nn.sigmoid(pre_activation, name='propagate_down')
        return pre_activation, post_activation

    def sample_hidden_given_visible(self, v_sample: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        h_pre, h_mean = self.propagate_up(v_sample)
        h_sample = self.sample(h_mean, tf.shape(h_pre))
        return h_pre, h_mean, h_sample

    def sample_visible_given_hidden(self, h_sample: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        v_pre, v_mean = self.propagate_down(h_sample)
        v_sample = self.sample(v_mean, tf.shape(v_pre))
        return v_pre, v_mean, v_sample

    def gibbs_vis_hid_vis(self, v_sample: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        h_pre, h_mean, h_sample = self.sample_hidden_given_visible(v_sample)
        v_pre, v_mean, v1_sample = self.sample_visible_given_hidden(h_sample)
        return h_pre, h_mean, h_sample, v_pre, v_mean, v1_sample

    def gibbs_hid_vis_hid(self, h_sample: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        v_pre, v_mean, v_sample = self.sample_visible_given_hidden(h_sample)
        h_pre, h_mean, h1_sample = self.sample_hidden_given_visible(v_sample)
        return v_pre, v_mean, v_sample, h_pre, h_mean, h1_sample

    def loss(self):
        sample = tf.divide(tf.cast(self.data_in.next()['image'], dtype=tf.float32), 255)
        sample = tf.reshape(sample, (self.batch_size, self.n_visible))
        with tf.GradientTape(persistent=True) as g:
            costs = tf.map_fn(lambda x: self.get_cost_updates(x), sample)
        grads = [g.gradient(costs, y) for y in self.params]
        return costs, grads

    def loss_p_dist(self):
        sample = self.data_in.next()
        sample = tf.cast(tf.reshape(sample, (self.batch_size, self.n_visible)), dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            costs = tf.map_fn(lambda x: self.get_cost_updates(x), sample)
        grads = [g.gradient(costs, y) for y in self.params]
        return costs, grads

    def get_cost_updates(self, sample: tf.Tensor, lr=0.1, persistent=None, k=1):
        h_pre, h_mean, h_sample = self.sample_hidden_given_visible(sample)
        v_pre, v_mean, v_sample = self.sample_visible_given_hidden(h_sample)

        if persistent is None:
            chain_start = h_sample
        else:
            chain_start = persistent

        nv_pre, nv_mean, nv_samples, nh_pre, nh_mean, nh_samples = self.gibbs_hid_vis_hid(h_sample)
        chain_end = nv_samples
        fe_sample = self.free_energy(sample)
        fe_end = self.free_energy(chain_end)
        cost = tf.subtract(fe_sample, fe_end)

        if persistent:
            persistent = nh_samples
            monitoring_cost = self.get_pseudo_likelihood(chain_end)
        else:
            monitoring_cost = self.get_reconstruction_cost(chain_end, nv_pre)

        return cost

    def get_pseudo_likelihood(self, sample):
        bit_i = tf.Variable(0, name='bit_i')
        xi = tf.math.round(sample)
        f_xi = self.free_energy(xi)
        xi_flip = copy.copy(xi)
        xi_flip[:, bit_i] = 1 - xi_flip[:, bit_i]
        f_xi_flip = self.free_energy(xi_flip)

        cost = tf.reduce_mean(self.n_visible * tf.math.log(tf.nn.sigmoid(f_xi_flip - f_xi)))
        sample[bit_i] = tf.math.floormod(tf.add(bit_i, 1), self.n_visible)
        return cost

    def get_reconstruction_cost(self, sample, nv_pre):
        cross_entropy = tf.reduce_sum(sample * tf.math.log(tf.nn.sigmoid(nv_pre)) +
                                                     tf.subtract(1,sample) * tf.math.log(1 - tf.nn.sigmoid(nv_pre)))

        return cross_entropy

    @staticmethod
    def sample(p, shape):
        return tf.where(tf.less(p, tf.random.uniform(shape, minval=0.0, maxval=1.0)), x=tf.zeros_like(p), y=tf.ones_like(p))
