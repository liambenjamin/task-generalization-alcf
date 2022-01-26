import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class LipschitzLayer(tf.keras.layers.Layer):

    def __init__(self, units=32, output_dim=10, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128):
        super(LipschitzLayer, self).__init__()
        self.units = units
        self.output_dim = output_dim
        self.beta = beta
        self.gamma_A = gamma_A
        self.gamma_W = gamma_W
        self.epsilon = epsilon
        self.sigma = sigma
        self.M_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.sigma)
        self.U_init = tf.keras.initializers.GlorotUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.D_init = tf.keras.initializers.GlorotUniform()

    def build(self, input_shape):
        self.M_A = self.add_weight(shape=(self.units, self.units), initializer=self.M_init, trainable=True, name='M_A')
        self.M_W = self.add_weight(shape=(self.units, self.units), initializer=self.M_init, trainable=True, name='M_W')
        self.U = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.U_init, trainable=True, name='U')
        self.bias = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='bias')
        self.D = self.add_weight(shape=(self.units, self.output_dim), initializer=self.D_init, trainable=True, name='D')
        self.bias_out = self.add_weight(shape=(self.output_dim, ), initializer=self.bias_init, trainable=True, name='bias_out')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def compose_A(self):
        A = (1-self.beta) * (self.M_A + tf.transpose(self.M_A)) + self.beta * (self.M_A - tf.transpose(self.M_A))
        A = A - self.gamma_A * tf.eye(self.units)
        return A

    def compose_W(self):
        W = (1-self.beta) * (self.M_W + tf.transpose(self.M_W)) + self.beta * (self.M_W - tf.transpose(self.M_W))
        W = W - self.gamma_W * tf.eye(self.units)
        return W

    def call(self, inputs):

        A = self.compose_A()
        W = self.compose_W()

        x = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))

        states = {}

        for i in range(0,inputs.shape[1]):
            h = K.dot(inputs[:,i,:], self.U)
            z = h + K.dot(x,W) + self.bias
            tanh_z = tf.keras.activations.tanh(z) # σ(z) = σ(Wh+Ux+b)
            Ah = K.dot(x, A)
            x = x + self.epsilon * (Ah + tanh_z) # h + \epsilon * [Ah + σ(Wh+Ux+b)]
            states['X{}'.format(i+1)] = x

        y = tf.matmul(x, self.D) + self.bias_out
        states['Y'] = tf.math.sigmoid(y) if self.output_dim == 1 else tf.keras.activations.softmax(y)

        return states

"""
Description: given network parameters, constructs a coadjoint trained model.

    - Note: will perform adjoint update if `lc_weights[1]=0.0`
"""
def makeCoadjointLipschitzRNN(hid_dim, ft_dim, out_dim, T, penalty_function, lc_weights, loss, optimizer, embed=False, **kwargs):

    if embed:
        # define layers
        inputs = tf.keras.Input( shape=(T,), name='input-layer')
        embed = tf.keras.layers.Embedding(5000, ft_dim, input_length=500) # output: (None, 500, 1, 50)
        rnn_step = LipschitzLayer(units=hid_dim, output_dim=out_dim, **kwargs)
        #
        vec_inputs = embed(inputs)
        outputs = rnn_step(vec_inputs)
    else:
        # define layers
        inputs = tf.keras.Input( shape=(T,ft_dim), name='input-layer')
        rnn_step = LipschitzLayer(units=hid_dim, output_dim=out_dim, **kwargs) # The output has shape `[bs, hid_dim]

        # outputs: dictionary of hidden states indexed by keys: "X1", "X2", ..., "X<arg T>"
        outputs = rnn_step(inputs)

    model = coadjointModel(
        inputs=inputs,
        outputs=outputs,
        name="lipschitz-rnn-coadjoint-model")

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function)

    return model
