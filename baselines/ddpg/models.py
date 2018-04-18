import tensorflow as tf
import tensorflow.contrib as tc

import tflearn
import numpy as np


def determine_shape(input_amount, kernel_size, padding, stride):
    return int((input_amount - kernel_size + 2 * padding) / stride + 1)


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, asset_features_shape=None):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.asset_features_shape = asset_features_shape
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            asset_inputs = tf.slice(obs,
                                    [0, 0],
                                    [-1, np.prod(self.asset_features_shape)])
            portfolio_inputs = tf.slice(obs,
                                        [0, np.prod(self.asset_features_shape)],
                                        [-1, self.nb_actions])
            portfolio_inputs = portfolio_inputs[:, 1:]
            asset_inputs = tf.reshape(asset_inputs,
                                      shape=[-1] + self.asset_features_shape)
            asset_inputs = asset_inputs[:, 1:, :, :]
            portfolio_inputs = tf.reshape(portfolio_inputs,
                                          [-1, self.nb_actions - 1, 1, 1])
            x = tc.layers.conv2d(inputs=asset_inputs,
                                 num_outputs=3,
                                 kernel_size=[1, 3],
                                 padding='VALID',
                                 activation_fn=None)

            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=10,
                                 kernel_size=[1, determine_shape(input_amount=self.asset_features_shape[1],
                                                                 kernel_size=3,
                                                                 padding=0,
                                                                 stride=1)],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tflearn.layers.merge_ops.merge([portfolio_inputs, x], 'concat', axis=-1)
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=1,
                                 kernel_size=[1, 1],
                                 padding='VALID',
                                 activation_fn=None)
            x = x[:, :, 0, 0]
            bias = tf.get_variable('bias_actor', [1, 1], dtype=tf.float32,
                                   initializer=tf.zeros_initializer)
            bias = tf.tile(bias, [tf.shape(x)[0], 1])
            x = tf.concat([bias, x], 1)
            x = tf.nn.softmax(x)
        return x


class Critic(Model):
    def __init__(self, nb_actions, name='critic', layer_norm=True, asset_features_shape=None):
        super(Critic, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.asset_features_shape = asset_features_shape

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            asset_inputs = tf.slice(obs,
                                    [0, 0],
                                    [-1, np.prod(self.asset_features_shape)])
            portfolio_inputs = tf.slice(obs,
                                        [0, np.prod(self.asset_features_shape)],
                                        [-1, self.nb_actions])
            portfolio_inputs = portfolio_inputs[:, 1:]
            asset_inputs = tf.reshape(asset_inputs,
                                      shape=[-1] + self.asset_features_shape)
            asset_inputs = asset_inputs[:, 1:, :, :]
            portfolio_inputs = tf.reshape(portfolio_inputs,
                                          [-1, self.nb_actions - 1, 1, 1])
            x = tc.layers.conv2d(inputs=asset_inputs,
                                 num_outputs=3,
                                 kernel_size=[1, 3],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=10,
                                 kernel_size=[1, determine_shape(input_amount=self.asset_features_shape[1],
                                                                 kernel_size=3,
                                                                 padding=0,
                                                                 stride=1)],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            auxil = tc.layers.conv2d(inputs=x,
                                     num_outputs=1,
                                     kernel_size=[1, 1],
                                     padding='VALID',
                                     activation_fn=None)
            auxil = tflearn.flatten(auxil)

            x = tflearn.layers.merge_ops.merge([portfolio_inputs, x], 'concat', axis=-1)
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=1,
                                 kernel_size=[1, 1],
                                 padding='VALID',
                                 activation_fn=None)
            x = x[:, :, 0, 0]
            bias = tf.get_variable('bias_critic', [1, 1], dtype=tf.float32,
                                   initializer=tf.zeros_initializer)
            bias = tf.tile(bias, [tf.shape(x)[0], 1])
            x = tf.concat([bias, x], 1)

            t1 = tflearn.fully_connected(x, 512)
            t2 = tflearn.fully_connected(action, 512)

            x = tf.add(t1, t2)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x, auxil

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
