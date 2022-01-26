import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, SimpleRNNCell

"""
Description: RNN layer class
"""
class RNNLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, output_dim=10):
        super().__init__()
        self.units = units
        self.output_dim = output_dim

    def build(self, input_shape):
        self.V = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True, name='V')
        self.W = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', trainable=True, name='W')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='bias')
        self.W_out = self.add_weight(shape=(self.units, self.output_dim), initializer='glorot_uniform', trainable=True, name='W_out')
        self.bias_out = self.add_weight(shape=(self.output_dim,), initializer='zeros', trainable=True, name='bias_out')
        self.x0 = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='initial_state')

    def call(self, inputs):
        """
        inputs: (batch, time_steps, ft_dim)
        """
        x = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        states = {}

        for i in range(0,inputs.shape[1]):
            h = K.dot(inputs[:,i,:], self.V)
            z = h + K.dot(x,self.W) + self.bias
            x = tf.keras.activations.tanh(z) # σ(z) = σ(Wh+Ux+b)
            states['X{}'.format(i+1)] = x

        y = tf.matmul(x, self.W_out) + self.bias_out
        states['Y'] = tf.math.sigmoid(y) if self.output_dim == 1 else tf.keras.activations.softmax(y)

        return states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'output_dim': self.output_dim
        })
        return config

"""
Description: RNN layer class
"""
class RNNLayer2(tf.keras.layers.Layer):
    def __init__(self, hid_dim, ft_dim, time_steps, activation, initializer):
        super().__init__()
        self.cell = tf.keras.layers.SimpleRNNCell(hid_dim, activation=activation, recurrent_initializer=initializer)
        self.hid_dim = hid_dim
        self.ft_dim = ft_dim
        self.time_steps = time_steps

    def call(self, inputs):
        """
        inputs: (batch, time_steps, ft_dim)
        """
        #bs = tf.shape(inputs)[0]
        input_split = tf.split(inputs, self.time_steps, axis=1) # list length time steps w/ element (batch, 1, ft_dim)
        states = {}
        #X0 = self.cell.get_initial_state(tf.squeeze(input_split[0], axis=1))
        #_, states["X1"] = self.cell(tf.squeeze(input_split[0], axis=1), X0)
        _, states["X1"] = self.cell(tf.squeeze(input_split[0], axis=1), self.cell.get_initial_state(batch_size=tf.shape(inputs)[0], dtype='float32'))
        for i in range(1,len(input_split)):
            _, states["X{}".format(i+1)] = self.cell(tf.squeeze(input_split[i], axis=1), states["X{}".format(i)])

        return states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'cell': self.cell,
            'hid_dim': self.hid_dim,
            'ft_dim': self.ft_dim,
            'time_steps': self.time_steps
        })
        return config


"""
Description: given network parameters, constructs a coadjoint trained model.

    - Note: will perform adjoint update if `lc_weights[1]=0.0`
"""
def makeCoadjointRNN(hid_dim, ft_dim, T, output_dim, penalty_function, lc_weights, loss, optimizer, embed=False):

    if embed:
        # layers
        inputs = tf.keras.Input( shape=(T,), name='input-layer')
        embed = tf.keras.layers.Embedding(5000, ft_dim, input_length=500) # output: (None, 500, 1, 50)
        rnn_step = RNNLayer(hid_dim, output_dim) # The output has shape `[bs, hid_dim]
        # output
        vec_inputs = embed(inputs)
        outputs = rnn_step(vec_inputs)

    else:
        # layers
        inputs = tf.keras.Input( shape=(T,ft_dim), name='input-layer')
        rnn_step = RNNLayer(hid_dim, output_dim) # The output has shape `[bs, hid_dim]
        # output
        outputs = rnn_step(inputs) # dictionary of hidden states

    # Model outputs reverse ordered:
    # output -> Y -> XT -> ... -> X1
    model = coadjointModel(
        inputs=inputs,
        outputs=outputs,
        name="rnn-tanh-coadjoint-model")

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function
                 )

    return model

"""
Description: given network parameters, constructs a coadjoint trained model.

    - Note: will perform adjoint update if `lc_weights[1]=0.0`

def makeCoadjointRNN(hid_dim, ft_dim, out_dim, T, activation, output_activation, penalty_function, lc_weights, loss, optimizer):

    # define layers
    inputs = keras.Input( shape=(T,ft_dim), name='input-layer')
    rnn_step = RNNLayer(hid_dim, ft_dim, T, activation, 'orthogonal') # The output has shape `[bs, hid_dim]
    dense_output = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')

    outputs = rnn_step(inputs) # dictionary of hidden states
    outputs["Y"] = dense_output(outputs["X{}".format(T)])

    # output -> Y -> XT -> ... -> X1
    model = coadjointModel(
        inputs=inputs,
        outputs=outputs,
        name="rnn-{0}-coadjoint-model".format(activation))

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function)

    return model
"""
