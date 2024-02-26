import os
import numpy as np
from train import train


def sim_f(H, persis_info, sim_specs, _):
    """
    Wraps model training function
    """
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ["KMP_BLOCKTIME"] = '0'
    os.environ["KMP_AFFINITY"]= 'granularity=fine,verbose,compact,1,0'

    # parse hyperparameters from H
    architecture = H['model_type'][0]
    learning_rate = H['learning_rate'][0]
    hid_dim = H['hid_dim'][0]
    epochs = H['epochs'][0]
    dataset = H['dataset'][0]
    permute = H['permute'][0]
    pad = H['pad'][0]
    orientation = H['orientation'][0]
    identifier = H['identifier'][0]
    sim_path = os.getcwd()


    # train model
    training_history = train(architecture, learning_rate, hid_dim, epochs, dataset, permute, pad, orientation, identifier)

    # generate output history array according to sim_specs
    H_out = np.zeros(1, dtype=sim_specs['out'])
    H_out['sim_path'] = sim_path


    return H_out, persis_info
