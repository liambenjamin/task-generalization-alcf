import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Embedding, Layer
import rnn, lstm, gru, antisymmetric, unitary, lipschitz, exponential, unicornn

"""
Custom gradient for tf.norm
    - L2 norm
"""
@tf.custom_gradient
def norm(x): #x (bs, hid_dim)
    ϵ = 1.0e-17
    nrm = tf.norm(x, axis=1, keepdims=True)
    def grad(dy):
        return dy * tf.math.divide(x,(nrm + ϵ))
    return nrm, grad

"""
Scaled Variance Adjoint Penalty Function
"""
@tf.function(experimental_relax_shapes=True)
def scaled_variance(adjoints):
    N = len(adjoints)
    λN = norm(adjoints["X{0}".format(N)]) #(batch_size, 1)

    m2 = tf.zeros(tf.shape(λN))
    scaled_var = tf.zeros(tf.shape(m2))

    for i in range(1,N+1):
        λ = norm(adjoints["X{0}".format(i)])
        m2 += λ

    m2 = m2 / N

    for i in range(1,N+1):
        λ = norm(adjoints["X{0}".format(i)]) #(batch_size,1)
        scaled_var += tf.math.pow(λ - m2, 2)

    return tf.reduce_mean(scaled_var)

"""
Scaled Variance Adjoint Penalty Function for complex hidden states
"""
@tf.function(experimental_relax_shapes=True)
def complex_scaled_variance(adjoints):
    N = len(adjoints)
    λN = norm(adjoints["X{0}".format(N)]) #(batch_size, 1)
    m2 = tf.zeros(tf.shape(λN), dtype=tf.dtypes.complex64)
    scaled_var = tf.zeros(tf.shape(m2), dtype=tf.dtypes.complex64)

    for i in range(1,N+1):
        λ = norm(adjoints["X{0}".format(i)])
        m2 += λ

    m2 = m2 / N

    for i in range(1,N+1):
        λ = norm(adjoints["X{0}".format(i)]) #(batch_size,1)
        scaled_var += tf.math.pow(λ - m2, 2)

    return tf.reduce_mean(scaled_var)


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
Adjoint Class Implementation
"""
class adjointModel(tf.keras.Model):

    def compile(self, optimizer, loss_fn, adj_penalty):
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.penalty_tracker = tf.keras.metrics.Mean(name='penalty')
        self.first_adjoint = tf.keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = tf.keras.metrics.Mean(name='last-adjoint')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        super(adjointModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        x, y = data

        with tf.GradientTape(persistent=True) as t1:
            outputs = self(x, training=True)
            L = self.loss_fn(y, outputs["Y"]) # (true label, prediction)
            #L += tf.reduce_sum(self.losses)
            preds = outputs.pop("Y")

        #Compute Adjoint-based Derivatives of Weights
        dL_dW = t1.gradient(L, self.trainable_variables)
        #Compute Adjoints
        dL_dX = t1.gradient(L, outputs)

        #Compute Adjoint Penalty
        G = self.adj_penalty( dL_dX )

        # Delete persistent tape
        del t1

        # for embedding layer compatabiity
        dL_dW[0] = tf.convert_to_tensor(dL_dW[0])

        # update parameters
        self.optimizer.apply_gradients(zip(dL_dW, self.trainable_variables))

        N = len(dL_dX)
        λ1 = tf.norm(dL_dX["X1"], axis=1)
        λN = tf.norm(dL_dX["X{0}".format(N)], axis=1)

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,preds)
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.penalty_tracker, self.accuracy_tracker,
                    self.first_adjoint, self.last_adjoint]

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        with tf.GradientTape() as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs["Y"]) # (true label, prediction)
            preds = outputs.pop("Y")

        #Compute Adjoints
        dL_dX = t1.gradient(L, outputs)

        #Compute Adjoint Penalty
        G = self.adj_penalty( dL_dX )
        # Updates the metrics tracking the loss
        N = len(dL_dX)
        λ1 = tf.norm(dL_dX["X1"], axis=1)
        λN = tf.norm(dL_dX["X{0}".format(N)], axis=1)

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,preds)
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }


def differential(f, A, E):
    """ Computes the differential of f at A when acting on E:  (df)_A(E) """
    n = A.shape[0]
    Z = tf.zeros((n,n))

    top = tf.concat([A, E], axis=1)
    bottom = tf.concat([Z, A], axis=1)
    M = tf.concat([top, bottom], axis=0)

    return f(M)[:n, n:]

def update_expA(model, grad_B, lr):

    η = lr * 0.1
    B = model.get_layer('exp_rnn_layer').B
    A = model.get_layer('exp_rnn_layer').A
    E = 0.5 * (tf.matmul(tf.transpose(grad_B), B) - tf.matmul(tf.transpose(B), grad_B))
    grad_A = tf.matmul(B, differential(tf.linalg.expm, tf.transpose(A), E))
    update = A + η * grad_A
    model.get_layer('exp_rnn_layer').A.assign(update)

    return

def torch_update_expA(model, grad_B, lr):
    η = lr * 0.1

    B = model.get_layer('exp_rnn_layer').B
    A = model.get_layer('exp_rnn_layer').A

    dexp = differential(tf.linalg.expm, tf.transpose(A), grad_B)
    d_skew = tf.linalg.band_part(dexp - tf.transpose(dexp), 0, -1)

    update = A - η * d_skew
    model.get_layer('exp_rnn_layer').A.assign(update)
    return

"""
Exponential RNN Coadjoint Class Implementation
"""
class adjointModelExpRNN(tf.keras.Model):

    def compile(self, optimizer, loss_fn, adj_penalty, embed=False):
        self.embed = embed
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.penalty_tracker = keras.metrics.Mean(name='penalty')
        self.first_adjoint = keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = keras.metrics.Mean(name='last-adjoint')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        super(adjointModelExpRNN, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        # compute B=e**A prior to each batch update
        self.get_layer('exp_rnn_layer').reset_parameters()

        x, y = data

        with tf.GradientTape(persistent=True) as t1:
            outputs = self(x, training=True)
            L = self.loss_fn(y, outputs["Y"]) #(true label, prediction)
            #L += tf.reduce_sum(self.losses)
            preds = outputs.pop("Y")

        dL_dW = t1.gradient(L, self.trainable_variables)
        dL_dX = t1.gradient(L, outputs) #adjoints

        G = self.adj_penalty( dL_dX )

        del t1

        # for embedding layer compatabiity
        dL_dW[0] = tf.convert_to_tensor(dL_dW[0])

        # update parameters -- updates B and not A. The update for B is not used as B=e**A is computed on each batch
        self.optimizer.apply_gradients(zip(dL_dW, self.trainable_variables))

        # apply gradient update to A
        if self.embed:
            update_expA(self, dL_dW[2], self.optimizer.learning_rate) # Prop 4.1 gradient
        else:
            update_expA(self, dL_dW[1], self.optimizer.learning_rate) # Prop 4.1 gradient

        self.get_layer('exp_rnn_layer').reset_parameters()

        N = len(dL_dX)
        λ1 = tf.norm(dL_dX["X1"], axis=1)
        λN = tf.norm(dL_dX["X{0}".format(N)], axis=1)

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,preds)
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.penalty_tracker, self.accuracy_tracker,
                    self.first_adjoint, self.last_adjoint]

    @tf.function
    def test_step(self, data):

        self.get_layer('exp_rnn_layer').reset_parameters()

        # Unpack the data
        x, y = data

        # Compute predictions
        with tf.GradientTape() as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs["Y"]) # (true label, prediction)
            preds = outputs.pop("Y")

        #Compute Adjoints
        dL_dX = t1.gradient(L, outputs)

        #Compute Adjoint Penalty
        G = self.adj_penalty( dL_dX )

        # Compute initial and terminal adjoints
        N = len(dL_dX)
        λ1 = tf.norm(dL_dX["X1"], axis=1)
        λN = tf.norm(dL_dX["X{0}".format(N)], axis=1)

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,preds)
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

"""
Make adjoint trained recurrent model where architecture is specified by 'name'
"""
def build_adjoint_model(name, units, ft_dim, T, output_dim, learning_rate, embed=False):

    optimizer = tf.keras.optimizers.get('adam')
    optimizer.learning_rate = learning_rate
    loss = tf.keras.losses.BinaryCrossentropy() if output_dim == 1 else tf.keras.losses.SparseCategoricalCrossentropy()
    penalty_function = complex_scaled_variance if name == 'unitary' else scaled_variance
    output_activation = 'sigmoid' if output_dim == 1 else 'softmax'

    rec_layer = {   'rnn': rnn.RNNLayer(units, output_dim),
                    'lstm': lstm.LSTMLayer(units, ft_dim, T),
                    'gru': gru.GRULayer(units, ft_dim, T),
                    'antisymmetric': antisymmetric.antiRNNLayer(units=units, ft_dim=ft_dim, time_steps=T, gamma=0.01, epsilon=0.01, sigma=4),
                    'unitary': unitary.UnitaryLayer(units=units, output_dim=output_dim),
                    'lipschitz': lipschitz.LipschitzLayer(units=units, output_dim=output_dim, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128),
                    'exponential': exponential.expRNNLayer(units=units, time_steps=T),
                    'unicornn': unicornn.UniCORNNLayer(units=units, ft_dim=ft_dim, output_dim=output_dim, epsilon=0.03, alpha=0.9, L=2)
                    }

    if embed:
        if name == 'unitary':
            inputs = tf.keras.Input( shape=(T,), name='input_layer', dtype=tf.dtypes.complex64)
        else:
            inputs = tf.keras.Input( shape=(T,), name='input_layer')
        embed = tf.keras.layers.Embedding(5000, ft_dim, input_length=500)
        rnn_step = rec_layer[name]
        vec_inputs = embed(inputs)
        outputs = rnn_step(vec_inputs)

        if name in ['lstm', 'gru', 'antisymmetric', 'exponential']:
            dense_output = tf.keras.layers.Dense(output_dim, activation=output_activation, name='output-layer')
            outputs["Y"] = dense_output(outputs["X{}".format(T)])
    else:
        if name == 'unitary':
            inputs = tf.keras.Input( shape=(T,ft_dim), name='input_layer', dtype=tf.dtypes.complex64)
        else:
            inputs = tf.keras.Input( shape=(T,ft_dim), name='input_layer')

        rnn_step = rec_layer[name]
        outputs = rnn_step(inputs)

        if name in ['lstm', 'gru', 'antisymmetric', 'exponential']:
            dense_init = tf.keras.initializers.GlorotNormal()
            dense_output = tf.keras.layers.Dense(output_dim, activation=output_activation, kernel_initializer=dense_init, name='output-layer')
            outputs["Y"] = dense_output(outputs["X{}".format(T)])

    if name == 'exponential':
        model = adjointModelExpRNN(inputs=inputs, outputs=outputs, name="{0}-adjoint-model".format(name))
    else:
        model = adjointModel(inputs=inputs, outputs=outputs, name="{0}-adjoint-model".format(name))

    # Compile Model
    model.compile(optimizer=optimizer,
        loss_fn=loss,
        adj_penalty=penalty_function
        )

    return model
