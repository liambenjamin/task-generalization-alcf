import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

"""
Description: GRU layer class
"""
class GRULayer(tf.keras.layers.Layer):
    def __init__(self, hid_dim, ft_dim, time_steps):
        super().__init__()
        self.cell = tf.keras.layers.GRUCell(hid_dim)
        self.hid_dim = hid_dim
        self.ft_dim = ft_dim
        self.time_steps = time_steps

    def call(self, inputs):
        """
        inputs: (batch, time_steps, ft_dim)
        """
        input_split = tf.split(inputs, self.time_steps, axis=1) # list length time steps w/ element (batch, 1, ft_dim)
        states = {}
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
def makeCoadjointGRU(hid_dim, ft_dim, out_dim, T, output_activation, penalty_function, lc_weights, loss, optimizer, embed=True):

    # Input -- "None" is for variable batch size
    inputs = keras.Input( shape=(T,ft_dim), name='input-layer')

    # define layers
    gru_step = GRULayer(hid_dim, ft_dim, T) # The output has shape `[bs, hid_dim]
    dense_output = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')

    outputs = gru_step(inputs) # dictionary of hidden states
    outputs["Y"] = dense_output(outputs["X{}".format(T)])

    # Model outputs reverse ordered:
    # output -> XT -> ... -> X1
    model = coadjointModel(
        inputs=inputs,
        outputs=outputs,
        name="gru-coadjoint-model")

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function)

    return model
