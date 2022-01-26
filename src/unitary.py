import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

"""
Modified ReLU activation
"""
def modified_relu(x, bias, ϵ=0.0):

    nrm = K.abs(x)
    M = tf.keras.activations.relu(nrm + bias) / (nrm + ϵ)
    m_relu = tf.cast(M, dtype=tf.dtypes.complex64) * x

    return m_relu

"""
Complex variable uniform initialization
"""
class ComplexUniformInit(tf.keras.initializers.Initializer):

    def __init__(self, minval=-1, maxval=1):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape, dtype=tf.dtypes.complex64, **kwargs):
        P = tf.random.uniform(shape=shape, minval=self.minval, maxval=self.maxval)
        return tf.complex(P,P)

    def get_config(self):
        return {'minval': self.minval, 'maxval': self.maxval}

"""
Complex glorot uniform initialization
"""
class ComplexGlorotInit(tf.keras.initializers.Initializer):

    def __init__(self):
        self.glorot = tf.keras.initializers.GlorotUniform()

    def __call__(self, shape, dtype=tf.dtypes.complex64, **kwargs):
        P = self.glorot(shape=shape)
        return tf.cast(P, dtype=tf.dtypes.complex64)

"""
Applies fixed permutation across columns of (d x d) complex matrix
"""

class PermutationMatrix(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(PermutationMatrix, self).__init__()
        self.units = units
        self.permutation = tf.random.shuffle(tf.range(start=0, limit=self.units, dtype=tf.int32))

    def call(self, inputs):
        return tf.gather(inputs, self.permutation, axis=1)


class UnitaryLayer(tf.keras.layers.Layer):

    def __init__(self, units=32, output_dim=10):
        super(UnitaryLayer, self).__init__()
        self.units = units
        self.output_dim = output_dim
        self.d_init = tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi)#ComplexUniformInit(minval=-np.pi, maxval=np.pi)
        self.r_init = ComplexUniformInit(minval=-1, maxval=1)
        self.V_init = ComplexGlorotInit()
        self.U_init = tf.keras.initializers.GlorotUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.x0_init = ComplexUniformInit(minval=-np.sqrt(3/(2*self.units)), maxval=np.sqrt(3/(2*self.units)))

    def build(self, input_shape):
        self.d1 = self.add_weight(shape=(self.units,), initializer=self.d_init, dtype=tf.dtypes.float32, trainable=True, name='d1')
        self.d2 = self.add_weight(shape=(self.units,), initializer=self.d_init, dtype=tf.dtypes.float32, trainable=True, name='d2')
        self.d3 = self.add_weight(shape=(self.units,), initializer=self.d_init, dtype=tf.dtypes.float32, trainable=True, name='d3')
        self.r1 = self.add_weight(shape=(self.units,), initializer=self.r_init, dtype=tf.dtypes.complex64, trainable=True, name='r1')
        self.r2 = self.add_weight(shape=(self.units,), initializer=self.r_init, dtype=tf.dtypes.complex64, trainable=True, name='r2')
        self.Pi = PermutationMatrix(units=self.units)
        self.V = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.V_init, dtype=tf.dtypes.complex64, trainable=True, name='V')
        self.U = self.add_weight(shape=(2*self.units, self.output_dim), initializer='glorot_uniform', trainable=True, name='U')
        self.bias = self.add_weight(shape=(self.units,), initializer=self.bias_init, dtype=tf.dtypes.float32, trainable=True, name='bias')
        self.bias_out = self.add_weight(shape=(self.output_dim,), initializer=self.bias_init, trainable=True, name='bias_out')
        self.activation = modified_relu
        self.x0 = self.add_weight(shape=(self.units,), initializer=self.x0_init, dtype=tf.dtypes.complex64, trainable=False, name='initial_state')

    def compose_D(self):
        D1 = tf.linalg.diag(tf.complex(tf.cos(self.d1), tf.sin(self.d1)))
        D2 = tf.linalg.diag(tf.complex(tf.cos(self.d2), tf.sin(self.d2)))
        D3 = tf.linalg.diag(tf.complex(tf.cos(self.d3), tf.sin(self.d3)))
        return (D1, D2, D3)

    def compose_R(self):
        rr1 = 2 * tf.tensordot(self.r1, tf.math.conj(self.r1), axes=0) / tf.linalg.norm(self.r1) ** 2
        R1 = tf.eye(self.r1.shape[0], dtype=tf.dtypes.complex64) - rr1
        rr2 = 2 * tf.tensordot(self.r2, tf.math.conj(self.r2), axes=0) / tf.linalg.norm(self.r2) ** 2
        R2 = tf.eye(self.r2.shape[0], dtype=tf.dtypes.complex64) - rr2
        return (R1, R2)

    def compose_W(self):
        # W = D3 R2 F^-1 D2 Π R1 F D1
        (D1, D2, D3) = self.compose_D()
        (R1, R2) = self.compose_R()

        t1 = K.dot(R1, tf.signal.fft(D1))
        t2 = self.Pi(t1)
        t3 = tf.signal.ifft(K.dot(D2, t2))
        t4 = K.dot(R2, t3)
        W = K.dot(D3, t4)
        return W

    def call(self, inputs):

        W = self.compose_W()

        x = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units), dtype=tf.dtypes.complex64)

        states = {}

        for i in range(0,inputs.shape[1]):

            h = K.dot(inputs[:,i,:], self.V)
            output = h + K.dot(x, W)
            output = self.activation(output, self.bias)
            x = output
            states['X{}'.format(i+1)] = x

        x_flat = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=1)
        y = tf.matmul(x_flat, self.U) + self.bias_out
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
Description: given network parameters, constructs a coadjoint trained model.

    - Note: will perform adjoint update if `lc_weights[1]=0.0`
"""
def makeCoadjointUnitaryRNN(hid_dim, ft_dim, out_dim, T, penalty_function, lc_weights, loss, optimizer):

    # Input -- "None" is for variable batch size
    inputs = tf.keras.Input( shape=(T,ft_dim), name='input-layer', dtype=tf.dtypes.complex64)

    # define layers
    rnn_step = UnitaryLayer(units=hid_dim, output_dim=out_dim) # The output has shape `[bs, hid_dim]

    # outputs: dictionary of hidden states indexed by keys: "X1", "X2", ..., "X<arg T>"
    outputs = rnn_step(inputs)

    model = coadjointModel(
        inputs=inputs,
        outputs=outputs,
        name="unitary-rnn-coadjoint-model")

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        lc_weights = lc_weights,
        adj_penalty=penalty_function)

    return model
