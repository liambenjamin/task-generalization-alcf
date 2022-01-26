import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

"""
Activation acting on time scale parameter (i.e. c_i)
"""
def mod_tanh(inputs):
    return 0.5 + 0.5 * tf.keras.activations.tanh(inputs/2)

"""
UniCORNN Layer -- assumes fixed layer depth=2)
"""
class UniCORNNLayer(tf.keras.layers.Layer):

    def __init__(self, units=32, ft_dim=1, output_dim=10, epsilon=0.03, alpha=0.9, L=2):
        super(UniCORNNLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.output_dim = output_dim
        self.w_init = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
        self.V_init = tf.keras.initializers.HeUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.c_init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        self.D_init = tf.keras.initializers.GlorotUniform()
        self.epsilon = epsilon
        self.alpha = alpha
        self.state_init = tf.keras.initializers.Zeros()
        self.L = L
        self.rec_activation = tf.keras.activations.tanh
        self.time_activation = mod_tanh

        self.w1 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True, name='w1')
        self.w2 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True, name='w2')
        #self.V1 = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.V_init, trainable=True)
        self.V1 = self.add_weight(shape=(self.ft_dim, self.units), initializer=self.V_init, trainable=True, name='V1')
        self.V2 = self.add_weight(shape=(self.units, self.units), initializer=self.V_init, trainable=True, name='V2')
        self.b1 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='b1')
        self.b2 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='b2')
        self.c1 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True, name='c1')
        self.c2 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True, name='c2')
        self.D = self.add_weight(shape=(4*self.units, self.output_dim), initializer=self.D_init, trainable=True, name='D')
        self.bias_out = self.add_weight(shape=(self.output_dim, ), initializer=self.bias_init, trainable=True, name='bias_out')
        """
    def build(self, input_shape):
        # for each layer:
            # w: (m,)
            # V: (p, m)
            # b: (m, )
            # c: (m, ) -- modulates ϵ (time-scale)
        self.w1 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True)
        self.w2 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True)
        self.V1 = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.V_init, trainable=True)
        self.V2 = self.add_weight(shape=(self.units, self.units), initializer=self.V_init, trainable=True)
        self.b1 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True)
        self.b2 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True)
        self.c1 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True)
        self.c2 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True)
        self.D = self.add_weight(shape=(4*self.units, self.output_dim), initializer=self.D_init, trainable=True)
        self.bias_out = self.add_weight(shape=(self.output_dim, ), initializer=self.bias_init, trainable=True)
        # initial states (y0, z0) -- set to 0-vector at run time
        """
    def call(self, inputs):

        y1 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        z1 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        y2 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        z2 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))

        states = {}

        for i in range(0,inputs.shape[1]):
            # 1. compute layer 1 z1
            # 2. compute layer 1 y1
            # 3. compute layer 2 z1
            # 4. compute layer 2 y2

            # layer 1: z1_nxt, y1_nxt
            δ1 = self.epsilon * self.time_activation(self.c1)
            h = K.dot(inputs[:,i,:], self.V1)
            h = self.rec_activation(h + tf.multiply(self.w1, y1) + self.b1)
            z1_nxt = z1 - δ1 * (h + self.alpha * y1)
            y1_nxt = y1 + δ1 * z1_nxt

            # layer 2: z2_nxt, y2_nxt
            δ2 = self.epsilon * self.time_activation(self.c2)
            h = K.dot(y1_nxt, self.V2)
            h = self.rec_activation(h + tf.multiply(self.w2, y2) + self.b2)
            z2_nxt = z2 - δ2 * (h + self.alpha * y2)
            y2_nxt = y2 + δ2 * z2_nxt

            # store states
            state_i = tf.concat([y1_nxt, z1_nxt, y2_nxt, z2_nxt], axis=1)
            states['X{0}'.format(i+1)] = state_i

            # reset states
            y1 = state_i[:, :self.units] # y1_nxt
            z1 = state_i[:, self.units:2*self.units] # z1_nxt
            y2 = state_i[:, 2*self.units:3*self.units] # y2_nxt
            z2 = state_i[:, 3*self.units:] # z2_nxt

        # map terminal state to output dimension
        # terminal state: (bs, 4 * self.units)
        out = tf.matmul(state_i, self.D) + self.bias_out
        states['Y'] = tf.math.sigmoid(out) if self.output_dim == 1 else tf.keras.activations.softmax(out)

        return states

def makeCoadjointUniCORNN(hid_dim, ft_dim, out_dim, T, penalty_function, lc_weights, loss, optimizer, **kwargs):

    # define layers
    inputs = tf.keras.Input( shape=(T,ft_dim), name='input-layer')
    rnn_step = UniCORNNLayer(units=hid_dim, ft_dim=ft_dim, output_dim=out_dim, **kwargs) # The output has shape `[bs, hid_dim]

    # outputs: dictionary of hidden states indexed by keys: "X1", "X2", ..., "X<arg T>"
    outputs = rnn_step(inputs)

    model = coadjointModel(
        inputs=inputs,
        outputs=outputs,
        name="unicornn-rnn-coadjoint-model")

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function)

    return model
