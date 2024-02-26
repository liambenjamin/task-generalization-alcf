import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense


class antiRNNLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, ft_dim=1, time_steps=None, epsilon=0.01, gamma=0.01, sigma=0.01):
        super(antiRNNLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma
        self.ft_dim = ft_dim
        self.time_steps = time_steps
        self.V = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.ft_dim),
            trainable=True, name='V'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=self.sigma/self.units),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')


    def call(self, inputs):

        states = {}
        x = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))

        for i in range(0, self.time_steps):

            M = self.W - tf.transpose(self.W) - self.gamma * tf.eye(self.units)
            h = K.dot(inputs[:,i,:], self.V)
            z = h + K.dot(x,M) + self.bias
            tanh_z = tf.keras.activations.tanh(z)
            x = x + self.epsilon * tanh_z
            states['X{}'.format(i+1)] = x

        return states
