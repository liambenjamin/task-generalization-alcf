"""
Import Libraries
"""
import sys
import uuid
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from network import build_adjoint_model
from network import TimeCallback
from load_data import load_experiment


def train_model(architecture, learning_rate, hid_dim, epochs, dataset, permute, pad, orientation):

    # generate experiment data
    (x_train, y_train), (x_test, y_test) = load_experiment(dataset,
                                                        permute=permute,
                                                        pad=pad,
                                                        orientation=orientation
                                                        )

    if architecture == 'unitary':
        x_train = tf.cast(x_train, dtype=tf.dtypes.complex64)
        x_test = tf.cast(x_test, dtype=tf.dtypes.complex64)

    identifier = uuid.uuid1()

    T = int(x_train.shape[1])
    ft_dim = 50 if dataset in ['imdb', 'reuters'] else int(x_train.shape[2])
    output_dim = 1 if dataset in ['imdb', 'adding'] else int(len(np.unique(y_train)))
    embed = True if dataset in ['imdb', 'reuters'] else False

    treatment = {'model_type': architecture, 'dataset': dataset, 'permute': permute,
                    'pad': pad, 'orientation': orientation, 'hid_dim': hid_dim,
                    'lc_weights': [1.0, 0.0], 'epochs': epochs, 'learning_rate': learning_rate,
                    'identifier': identifier
                    }

    # training callback(s)
    time_callback = TimeCallback()

    print('\nTreatment:\n', treatment, '\n')

    # build model
    model = build_adjoint_model(architecture, hid_dim, ft_dim, T, output_dim, learning_rate, embed=embed)

    print('starting training...\n')
    # fit model
    model_out = model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=epochs, verbose=1, callbacks=[time_callback])

    # store model/training information in history
    model_out.history['time'] = time_callback.times
    model_out.history['treatment'] = treatment

    return model_out.history


def main():
    params_file = sys.argv[1]
    read_f = open(params_file)
    hypers = json.load(read_f)

    arch = hypers['architecture']
    learning_rate = hypers['learning_rate']
    hid_dim = hypers['dimension']
    epochs = hypers['epochs']
    dataset = hypers['dataset']
    permute = hypers['permute']
    pad = hypers['pad']
    orientation = hypers['orientation']


    output = train_model(arch, learning_rate, hid_dim, epochs, dataset, permute, pad, orientation)
    print('\ntraining output:\n', output)

if __name__== "__main__":
    main()
