#!/usr/bin/env python
import os
import requests
import datetime
import numpy as np
from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams, save_libE_output
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f # for no generator
from libensemble import logger
from training_dependencies import generate_test_arguments
from simulator import sim_f # training function


logger.set_level('DEBUG')

# Parse comms type, number of workers, etc. from command-line
nworkers, is_manager, libE_specs, _ = parse_args()

sim_specs = {'sim_f': sim_f,
             'in': ['model_type', 'learning_rate', 'hid_dim', 'epochs', 'dataset', 'permute', 'pad', 'orientation', 'identifier'],
             'out': [('sim_path', '<U70')]
                        }
# Parameterize our persistent generator function
gen_specs = {} #empty b/c input arguments are independent of simulator

# Generate training arguments
training_args = generate_test_arguments()
n_samp = len(training_args['model_type'])


H0 = np.zeros(n_samp, dtype=[ ('model_type', '<U13'), ('learning_rate', float),
                                ('hid_dim', int), ('epochs', int),
                                ('dataset', '<U17'), ('permute', bool),
                                ('pad', int), ('orientation', '<U7'),
                                ('identifier', '<U40'), ('sim_id', int),
                                ('given', bool) ])

H0['model_type'] = training_args['model_type']
H0['learning_rate'] = training_args['learning_rate']
H0['hid_dim'] = training_args['hid_dim']
H0['epochs'] = training_args['epochs']
H0['dataset'] = training_args['dataset']
H0['permute'] = training_args['permute']
H0['pad'] = training_args['pad']
H0['orientation'] = training_args['orientation']
H0['identifier'] = training_args['identifier']
H0['sim_id'] = range(n_samp)
H0['given'] = False

alloc_specs = {'alloc_f': alloc_f, 'out': [('model_type', '<U13'), ('learning_rate', float),
                ('hid_dim', int), ('epochs', int), ('dataset', '<U17'),
                ('permute', bool), ('pad', int), ('orientation', '<U7'),
                ('identifier', '<U40')]}

exit_criteria = {'sim_max': len(H0)}
persis_info = add_unique_random_streams({}, nworkers + 1)

# Additional libEnsemble settings to customize our ensemble directory
libE_specs['ensemble_dir_path'] = './ensemble'

# Call libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info=persis_info,
                            alloc_specs=alloc_specs,
                            libE_specs=libE_specs,
                            H0=H0
                            )

if is_manager:
    assert len(H) == len(H0)
    assert np.array_equal(H0['model_type'], H['model_type'])
    assert np.array_equal(H0['learning_rate'], H['learning_rate'])
    assert np.array_equal(H0['hid_dim'], H['hid_dim'])
    assert np.array_equal(H0['epochs'], H['epochs'])
    assert np.array_equal(H0['dataset'], H['dataset'])
    assert np.array_equal(H0['permute'], H['permute'])
    assert np.array_equal(H0['pad'], H['pad'])
    assert np.array_equal(H0['orientation'], H['orientation'])
    assert np.all(H['returned'])
    print("\nlibEnsemble correctly didn't add anything to initial sample")
    save_libE_output(H, persis_info, __file__, nworkers)
