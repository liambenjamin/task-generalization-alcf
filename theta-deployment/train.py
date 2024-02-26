import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import training_dependencies
from training_dependencies import make_model
from training_dependencies import TimeCallback
from load_data import load_experiment


def train(architecture, learning_rate, hid_dim, epochs, dataset, permute, pad, orientation, identifier):

    # fix adjoint training
    penalty_weight = 0.0

    # generate experiment data
    (x_train, y_train), (x_test, y_test) = load_experiment(dataset, permute, pad, orientation)

    T = int(x_train.shape[1])
    ft_dim = 50 if dataset in ['imdb.npz', 'reuters.npz'] else int(x_train.shape[2])
    output_dim = 1 if dataset in ['imdb.npz', 'adding'] else int(len(np.unique(y_train)))
    embed = True if dataset in ['imdb.npz', 'reuters.npz'] else False

    time_callback = TimeCallback()
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    # build model
    model = make_model(architecture, T, ft_dim, hid_dim, output_dim, penalty_weight, learning_rate, embed=embed)

    # fit model
    model_out = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test,y_test), callbacks=[time_callback, nan_callback])

    # store model/training information in history
    model_out.history['model_type'] = architecture
    model_out.history['dataset'] = dataset
    model_out.history['permute'] = permute
    model_out.history['pad'] = pad
    model_out.history['orientation'] = orientation
    model_out.history['hid_dim'] = hid_dim
    model_out.history['penalty_weight'] = penalty_weight
    model_out.history['epochs'] = epochs
    model_out.history['learning_rate'] = learning_rate
    model_out.history['epoch_time'] = time_callback.times
    model_out.history['num_parameters'] = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
    model_out.history['identifier'] = identifier
    nan_flag = np.isnan(model_out.history['loss'][-1])
    model_out.history['nan'] = nan_flag

    np.save(f'training-history-{identifier}', model_out.history)

    return model_out.history
