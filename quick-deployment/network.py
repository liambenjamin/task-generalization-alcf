import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Embedding, Layer
from architectures import BasicRNN, LSTM, GRU, AntisymmetricRNN, LipschitzRNN, ExponentialRNN, UnICORNN


"""
Callback: Collect Epoch Training Times
"""
class TimeCallback(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


"""
Keras Adjoint Model Class
"""
class AdjointModel(tf.keras.Model):

    def compile(self, model_name, optimizer, loss_fn, embed=False):
        self.model_name = model_name
        self.embed = embed
        self.loss_fn = loss_fn
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        
        super(AdjointModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        # compute B=e**A prior to each batch update for Exponential RNN
        if self.model_name == 'exponential':
            self.get_layer('exponential_rnn').reset_parameters()

        x, y = data

        with tf.GradientTape() as tape:
          
            outputs = self(x, training=True)
            L = self.loss_fn(y, outputs[0])

        dL_dW = tape.gradient(L, self.trainable_variables)
        
        # for embedding layer compatabiity
        dL_dW[0] = tf.convert_to_tensor(dL_dW[0])
        
        # update parameters
        self.optimizer.apply_gradients(zip(dL_dW, self.trainable_variables))

        # update for Exponential RNN
        if self.model_name == 'exponential':
            # apply gradient update to A
            if self.embed:
                update_expA(self, dL_dW[2], self.optimizer.learning_rate) # Prop 4.1 gradient
            else:
                update_expA(self, dL_dW[1], self.optimizer.learning_rate) # Prop 4.1 gradient

            self.get_layer('exponential_rnn').reset_parameters()


        #Update Metrics
        self.loss_tracker.update_state(L)
        self.accuracy_tracker.update_state(y,outputs[0])

        return {"loss": self.loss_tracker.result(),
                "accuracy": self.accuracy_tracker.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        outputs = self(x)
        L = self.loss_fn(y, outputs[0]) # (true label, prediction)

        self.loss_tracker.update_state(L)
        self.accuracy_tracker.update_state(y,outputs[0])
        
        return {"loss": self.loss_tracker.result(),
                "accuracy": self.accuracy_tracker.result()
                }

"""
Computes the differential of f at A when acting on E:  (df)_A(E)
    - helper for computing gradient of Exponential RNN
"""
def differential(f, A, E):
    n = A.shape[0]
    Z = tf.zeros((n,n))
    top = tf.concat([A, E], axis=1)
    bottom = tf.concat([Z, A], axis=1)
    M = tf.concat([top, bottom], axis=0)

    return f(M)[:n, n:]

"""
Computes update for Exponential RNN
"""
def update_expA(model, grad_B, lr):
    η = lr * 0.1
    B = model.get_layer('exponential_rnn').B
    A = model.get_layer('exponential_rnn').A
    E = 0.5 * (tf.matmul(tf.transpose(grad_B), B) - tf.matmul(tf.transpose(B), grad_B))
    grad_A = tf.matmul(B, differential(tf.linalg.expm, tf.transpose(A), E))
    update = A + η * grad_A
    model.get_layer('exponential_rnn').A.assign(update)

    return

"""
Creates recurrent model from provided arguments
"""
def build_recurrent_model(name, T, ft_dim, hid_dim, out_dim, learning_rate, embed=False):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # fixed to adam
    batch_size = 32 # fixed to 32

    output_activation = 'softmax' if out_dim > 1 else 'sigmoid'
    loss = tf.keras.losses.SparseCategoricalCrossentropy() if out_dim > 1 else tf.keras.losses.BinaryCrossentropy()

    layers = {'rnn': BasicRNN(hid_dim),
              'lstm': LSTM(hid_dim),
              'gru': GRU(hid_dim),
              'antisymmetric': AntisymmetricRNN(hid_dim, ft_dim, epsilon=0.01, gamma=0.01, sigma=0.01),
              'lipschitz': LipschitzRNN(hid_dim, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128),
              'exponential': ExponentialRNN(hid_dim),
              'unicornn': UnICORNN(hid_dim, ft_dim, epsilon=0.03, alpha=0.9, L=2)
             }

    if embed:
        inputs = tf.keras.Input( shape=(T,), batch_size=32, name='input-layer')
        embed = tf.keras.layers.Embedding(20000, ft_dim, input_length=T)
        rec_layer = layers[name]
        dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')
        vec_inputs = embed(inputs)
        states = rec_layer(vec_inputs)
        outputs = dense_layer(states[-1])

    else:    
        inputs = tf.keras.Input( shape=(T,ft_dim), batch_size=batch_size, name='input-layer')
        rec_layer = layers[name] 
        dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer', trainable=True)         
        states = rec_layer(inputs)
        outputs = dense_layer(states[-1])

    model = AdjointModel(inputs=inputs, outputs=[outputs, states])
    model.compile(optimizer=optimizer,loss_fn=loss,model_name=name,embed=embed)

    return model