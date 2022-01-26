import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

def create_diag_(A, diag):
    n = A.shape[0]
    diag_z = np.zeros(n-1)
    diag_z[::2] = diag
    A_init = tf.linalg.diag(diag_z, k=1)
    A_init = A_init - tf.transpose(A_init)
    return A_init

def cayley_init_(A):
    size = A.shape[0] // 2
    diag = tf.random.uniform(shape=(size,), minval=0., maxval=np.pi / 2.)
    diag = -tf.sqrt( (1. - tf.cos(diag)) / (1. + tf.cos(diag)) )

    return create_diag_(A, diag)

class cayleyInit(tf.keras.initializers.Initializer):

    def __init__(self, mean=0, stddev=1):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        A = tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
        A_cayley = cayley_init_(A)
        return tf.cast(A_cayley, tf.float32)

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}


class modrelu(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(modrelu, self).__init__()
        self.dim = dim

    def build(self, inputs):
        self.bias = tf.Variable(tf.random.uniform(shape=(self.dim,), minval=-0.01, maxval=0.01), trainable=True, name='bias')

    def call(self, inputs):
        nrm = tf.abs(inputs)
        biased_nrm = nrm + self.bias
        magnitude = tf.keras.activations.relu(biased_nrm)
        phase = tf.sign(inputs)
        return phase * magnitude

    def get_config(self):
        base_config = super(modrelu, self).get_config()
        config = {'bias': self.bias}
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class expRNNLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, time_steps=None):
        super(expRNNLayer, self).__init__()
        self.units = units
        self.time_steps = time_steps

    def build(self, input_shape):

        self.T = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1/input_shape[-1]),
            trainable=True,
            name='T'
        )

        self.A = self.add_weight(
            shape=(self.units, self.units),
            initializer = cayleyInit,
            trainable=False,
            name='A'
        )

        self.B = tf.Variable(tf.linalg.expm(self.A), trainable=True)

        self.activation = modrelu(self.units)

        self.h0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')


    def reset_parameters(self):
        # retraction to tangent space
        A = tf.linalg.band_part(self.A, 0, -1) # upper triangular matrix
        A = A - tf.transpose(A)
        self.A.assign(A)
        # assign B from retraction
        self.B.assign(tf.linalg.expm(self.A))


    def call(self, inputs): #[batch, T, p]

        states = {}
        x = tf.ones(shape=(tf.shape(inputs)[0], self.units)) * self.h0

        for i in range(0, self.time_steps):

            h = K.dot(inputs[:,i,:], self.T)
            h = h + K.dot(x, self.B)
            h = self.activation(h)

            states['X{}'.format(i+1)] = h
            x = h

        return states

"""
Description: given network parameters, constructs a coadjoint trained model.

    - Note: will perform adjoint update if `lc_weights[1]=0.0`
"""
def makeCoadjointExpRNN(hid_dim, ft_dim, out_dim, T, output_activation, penalty_function, lc_weights, loss, optimizer, embed=False):

    if embed:
        # layers
        inputs = tf.keras.Input( shape=(T,), name='input-layer')
        embed = tf.keras.layers.Embedding(5000, ft_dim, input_length=500) # output: (None, 500, 1, 50)
        rnn_step = expRNNLayer(units=hid_dim, time_steps=T)
        dense_output = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')

        # output
        vec_inputs = embed(inputs)
        outputs = rnn_step(vec_inputs)
        outputs["Y"] = dense_output(outputs["X{}".format(T)])

    else:
        # Input -- "None" is for variable batch size
        inputs = tf.keras.Input( shape=(T,ft_dim), name='input-layer')
        # define layers
        rnn_step = expRNNLayer(units=hid_dim, time_steps=T)
        dense_output = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')

        # forward
        outputs = rnn_step(inputs)
        outputs["Y"] = dense_output(outputs["X{}".format(T)])

    # Model outputs reverse ordered:
    # output -> XT -> ... -> X1
    model = coadjointModelExpRNN(
        inputs=inputs,
        outputs=outputs,
        name="expRNN-coadjoint-model")

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function
        )

    return model
