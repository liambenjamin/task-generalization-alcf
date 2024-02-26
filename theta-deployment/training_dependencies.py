import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import time, itertools, uuid, re

import architectures

"""
Generates training arguments for testing processes.
"""
def generate_test_arguments():

    N = 6

    model_type = np.empty(shape=(N,), dtype='<U13')
    learning_rate = np.empty(shape=(N,), dtype=float)
    hid_dim = np.empty(shape=(N,), dtype=int)
    epochs = np.empty(shape=(N,), dtype=int)
    dataset = np.empty(shape=(N,), dtype='<U17')
    permute = np.empty(shape=(N,), dtype=bool)
    pad = np.empty(shape=(N,), dtype=int)
    orientation = np.empty(shape=(N,), dtype='<U7')
    identifier = np.empty(shape=(N,), dtype='<U40')

    for i in range(0,N):
        learning_rate[i] = 0.0001
        epochs[i] = 3
        hid_dim[i] = 128
        dataset[i] = 'reuters.npz'
        permute[i] = False
        pad[i] = 0
        orientation[i] = 'None'
        identifier[i] = str(uuid.uuid4())
    
    model_type[0] = 'rnn'
    model_type[1] = 'rnn'
    model_type[2] = 'antisymmetric'
    model_type[3] = 'antisymmetric'
    model_type[4] = 'lipschitz'
    model_type[5] = 'lipschitz'

    args_dict = {'model_type': model_type,
                 'learning_rate': learning_rate,
                 'hid_dim': hid_dim,
                 'epochs': epochs,
                 'dataset': dataset,
                 'permute': permute,
                 'pad': pad,
                 'orientation': orientation,
                 'identifier': identifier
                 }

    return args_dict


"""
Generates experiment arguments for loading experimental batches of architectures, hyperparameters and tasks.
"""
def generate_libE_arguments(architectures, learning_rates, dims, epochs, datasets, permute, pad, orientation):
    """Returns list of  all treatment combinations of `architecture_name`"""

    argument_set = [architectures, learning_rates, dims, epochs, datasets, permute, pad, orientation]
    total_sets = list(itertools.product(*argument_set))

    trim_set = []

    for config in total_sets:

        pad, orientation = config[6], config[7]

        # remove non-conforming tasks (pad 0 w/ orientation and pad 1000 w/ none orientation)
        if pad == 0 and orientation != 'None' or pad == 1000 and orientation == 'None':
            continue
        else:
            trim_set.append(config)

    model_type = np.empty(shape=(len(trim_set),), dtype='<U13')
    learning_rate = np.empty(shape=(len(trim_set),), dtype=float)
    hid_dim = np.empty(shape=(len(trim_set),), dtype=int)
    epochs = np.empty(shape=(len(trim_set),), dtype=int)
    dataset = np.empty(shape=(len(trim_set),), dtype='<U17')
    permute = np.empty(shape=(len(trim_set),), dtype=bool)
    pad = np.empty(shape=(len(trim_set),), dtype=int)
    orientation = np.empty(shape=(len(trim_set),), dtype='<U7')


    for i, config in enumerate(trim_set):

        # [architectures, learning_rates, dims, epochs, datasets, permute, pad, orientation]
        model_type[i], learning_rate[i], hid_dim[i], epochs[i], dataset[i], permute[i], pad[i], orientation[i] = config

    # repeat arrays 10x (for each replicate of experiment)
    n_times = 10

    args_dict = {'model_type': np.tile(model_type, n_times),
                 'learning_rate': np.tile(learning_rate, n_times),
                 'hid_dim': np.tile(hid_dim, n_times),
                 'epochs': np.tile(epochs, n_times),
                 'dataset': np.tile(dataset, n_times),
                 'permute': np.tile(permute, n_times),
                 'pad': np.tile(pad, n_times),
                 'orientation': np.tile(orientation, n_times)}

    total_n = len(args_dict['model_type'])
    identifier = np.empty(shape=(total_n,), dtype='<U40')

    for i in range(0,len(identifier)):
        identifier[i] = str(uuid.uuid4())

    args_dict['identifier'] = identifier

    assert len(args_dict['hid_dim']) == len(args_dict['identifier']), 'argument lengths do not match.'

    return args_dict


"""
Given command line arguments (sys.argv[1:]), casts arguments to appropriate type
"""
def cast_arguments(vals):

    assert len(vals) == 9, 'Invalid number of arguments passed.'

    arch = str(vals[0])
    learning_rate = float(vals[1])
    penalty = float(vals[2])
    hid_dim = int(vals[3])
    epochs = int(vals[4])
    dataset = str(vals[5])
    permute = True if vals[6] == 'True' else False
    pad = int(vals[7])
    orientation = None if vals[8] == 'None' else vals[8]

    return (arch, learning_rate, penalty, hid_dim, epochs, dataset, permute, pad, orientation)

"""
Custom gradient for L2 norm
"""
@tf.custom_gradient
def norm(x): #x (bs, hid_dim)
    ϵ = 1.0e-17
    nrm = tf.norm(x, axis=1, keepdims=True)
    def grad(dy):
        return dy * tf.math.divide(x,(nrm + ϵ))
    return nrm, grad

"""
Scaled variance adjoint penalty
"""
@tf.function
def scaled_variance_adjoint_penalty(adjoints):

    tf.stack(adjoints) 
    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    t1 = tf.zeros(nrms[0].shape)
    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N
    G = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)


"""
Training Callback: Collect Epoch Training Times
"""
class TimeCallback(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


"""
Computes the differential of f at A when acting on E:  (df)_A(E)
"""
def differential(f, A, E):
    n = A.shape[0]
    Z = tf.zeros((n,n))
    top = tf.concat([A, E], axis=1)
    bottom = tf.concat([Z, A], axis=1)
    M = tf.concat([top, bottom], axis=0)

    return f(M)[:n, n:]


"""
Update specific to Exponential RNN
"""
def update_expA(model, grad_B, lr):
    η = lr * 0.1
    B = model.get_layer('exponential_rnn_layer').B
    A = model.get_layer('exponential_rnn_layer').A
    E = 0.5 * (tf.matmul(tf.transpose(grad_B), B) - tf.matmul(tf.transpose(B), grad_B))
    grad_A = tf.matmul(B, differential(tf.linalg.expm, tf.transpose(A), E))
    update = A + η * grad_A
    model.get_layer('exponential_rnn_layer').A.assign(update)

    return


"""
Keras Model Class Implementation
"""
class CoadjointModel(tf.keras.Model):

    def compile(self, model_name, optimizer, loss_fn, adj_penalty, lc_weights, embed=False):
        self.model_name = model_name
        self.embed = embed
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.lc_weights = lc_weights
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.penalty_tracker = tf.keras.metrics.Mean(name='penalty')
        self.first_adjoint = tf.keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = tf.keras.metrics.Mean(name='last-adjoint')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        super(CoadjointModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        # compute B=e**A prior to each batch update
        if self.model_name == 'exponential':
            self.get_layer('exponential_rnn_layer').reset_parameters()

        #self.layers[2].reset_parameters() if self.embed else self.layers[1].reset_parameters()

        x, y = data

        with tf.GradientTape() as t2:
            with tf.GradientTape(persistent=True) as t1:
                outputs = self(x, training=True)
                L = self.loss_fn(y, outputs[0])

            dL_dW = t1.gradient(L, self.trainable_variables)
            dL_dX = t1.gradient(L, outputs[1])

            G = self.adj_penalty( dL_dX )

        dG_dW = t2.gradient(G, self.trainable_variables)

        del t1

        # for embedding layer compatabiity
        if self.embed:
            dL_dW[0] = tf.convert_to_tensor(dL_dW[0])
            dG_dW[0] = tf.convert_to_tensor(dG_dW[0])

        dL_plus_G_dW = [
            tf.add( x[0] * self.lc_weights[0], x[1] * self.lc_weights[1] )
            for x in zip(dL_dW, dG_dW)
            ]

        # update parameters
        self.optimizer.apply_gradients(zip(dL_plus_G_dW, self.trainable_variables))


        if self.model_name == 'exponential':
            # apply gradient update to A
            if self.embed:
                update_expA(self, dL_plus_G_dW[2], self.optimizer.learning_rate) # Prop 4.1 gradient
            else:
                update_expA(self, dL_plus_G_dW[1], self.optimizer.learning_rate) # Prop 4.1 gradient

            self.get_layer('exponential_rnn_layer').reset_parameters()


        λ1 = tf.reduce_mean(tf.norm(dL_dX[0]))
        λN = tf.reduce_mean(tf.norm(dL_dX[-1]))

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
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
            L = self.loss_fn(y, outputs[0]) # (true label, prediction)

        #Compute Adjoints
        dL_dX = t1.gradient(L, outputs[1])

        #Compute Adjoint Penalty
        G = self.adj_penalty( dL_dX )

        #Update Metrics
        λ1 = tf.reduce_mean(norm(dL_dX[0]))
        λN = tf.reduce_mean(norm(dL_dX[-1]))
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

"""
Creates keras model from provided arguments
"""
def make_model(name, T, ft_dim, hid_dim, out_dim, penalty_weight, learning_rate, embed=False):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if out_dim > 1:
        output_activation = 'softmax'
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        output_activation = 'sigmoid'
        loss = tf.keras.losses.BinaryCrossentropy()

    layers = {'rnn': architectures.BasicRNN(hid_dim),
              'lstm': architectures.LSTM(hid_dim),
              'gru': architectures.GRU(hid_dim),
              'antisymmetric': architectures.AntisymmetricRNN(hid_dim, ft_dim, epsilon=0.01, gamma=0.01, sigma=0.01),
              'lipschitz': architectures.LipschitzRNN(hid_dim, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128),
              'exponential': architectures.ExponentialRNN(hid_dim),
              'unicornn': architectures.UnICORNN(hid_dim, ft_dim, epsilon=0.03, alpha=0.9, L=2)
             }

    if embed:
        penalty = scaled_variance_adjoint_penalty
        inputs = tf.keras.Input( shape=(T,), batch_size=32, name='input-layer')
        embed = tf.keras.layers.Embedding(20000, ft_dim, input_length=T)
        rec_layer = layers[name]
        dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')
        vec_inputs = embed(inputs)
        states = rec_layer(vec_inputs)
        outputs = dense_layer(states[-1])

    else:
        penalty = scaled_variance_adjoint_penalty
        inputs = tf.keras.Input( shape=(T,ft_dim), batch_size=32, name='input-layer')
        rec_layer = layers[name]
        dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer')
        states = rec_layer(inputs)
        outputs = dense_layer(states[-1])

    model = CoadjointModel(inputs=inputs,outputs=[outputs, states])

    # Compile Model
    model.compile(optimizer=optimizer,
                loss_fn=loss,
                lc_weights=[1.0, penalty_weight],
                adj_penalty=penalty,
                model_name=name,
                embed=embed
                )
    
    return model
