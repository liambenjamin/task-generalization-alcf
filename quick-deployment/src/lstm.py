import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM

def makeChronoLSTM(hid_dim, out_dim, T, ft_dim, output_activation, single_feature, optimizer, loss):
    # initialize model
    model = tf.keras.Sequential()
    model.add(LSTM(hid_dim, input_shape=(T,ft_dim), bias_initializer='zeros', unit_forget_bias=False))
    model.add(Dense(out_dim, activation=output_activation))
    # initialize weights
    # ordering: i,f,c,o
    out = model(single_feature)
    weights = model.get_weights() # input, forget, cell, output
    biases = weights[2]
    #b_i = biases[:hid_dim]
    #b_f = biases[hid_dim:hid_dim*2]
    #b_c = biases[hid_dim*2:hid_dim*3]
    #b_o = biases[hid_dim*3:]

    # forget bias
    bias_f = tf.math.log(tf.random.uniform((hid_dim,), minval=1, maxval=T))
    biases[hid_dim:hid_dim*2] = bias_f
    biases[:hid_dim] = -bias_f

    weights[2] = biases
    model.set_weights(weights)

    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

    return model

def makeLSTM(hid_dim, out_dim, T, ft_dim, output_activation, optimizer, loss):
    # initialize model
    model = tf.keras.Sequential()
    model.add(LSTM(hid_dim, input_shape=(T,ft_dim), bias_initializer='zeros'))
    model.add(Dense(out_dim, activation=output_activation))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


"""
Description: LSTM layer class
"""
class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, hid_dim, ft_dim, time_steps):
        super().__init__()
        self.cell = tf.keras.layers.LSTMCell(hid_dim)
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
        cell_states = {}
        _, [ states["X1"], cell_states["X1"] ] = self.cell(tf.squeeze(input_split[0], axis=1), self.cell.get_initial_state(batch_size=tf.shape(inputs)[0], dtype='float32'))
        for i in range(1,len(input_split)):
            _, [ states["X{}".format(i+1)], cell_states["X{}".format(i+1)] ] = self.cell(tf.squeeze(input_split[i], axis=1), [states["X{}".format(i)], cell_states["X{}".format(i)]])

        return states #, cell_states

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
def makeCoadjointLSTM(hid_dim, ft_dim, out_dim, T, output_activation, penalty_function, lc_weights, loss, optimizer):

    # Input -- "None" is for variable batch size
    inputs = keras.Input( shape=(T,ft_dim), name='input-layer')

    # define layers
    lstm_step = LSTMLayer(hid_dim, ft_dim, T) # The output has shape `[bs, hid_dim]
    dense_output = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')

    outputs, cell_states = lstm_step(inputs) # dictionary of hidden states
    outputs["Y"] = dense_output(outputs["X{}".format(T)])

    # Model outputs reverse ordered:
    # output -> XT -> ... -> X1
    model = coadjointModel(
        inputs=inputs,
        outputs=outputs,
        name="lstm-coadjoint-model")

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function)

    return model
