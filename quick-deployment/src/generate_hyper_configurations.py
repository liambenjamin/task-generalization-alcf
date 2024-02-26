import json
import itertools


arch = ['rnn', 'lstm', 'antisymmetric', 'exponential', 'unitary']
learning_rate = [1e-4, 1e-3]
dataset = ['mnist', 'fashion_mnist']
dim = [128, 256]
epochs = [25]
permute = [True]
pad = [0]
orientation = [None]

hypers = [arch, learning_rate, dim, epochs, dataset, permute, pad, orientation]

hyper_list = [list(itertools.product(*hypers))][0]

for i in range(0, len(hyper_list)):
        params = hyper_list[i]

        hyper_dict = {'architecture' : params[0], 'learning_rate' : params[1],
                'dimension' : params[2], 'epochs' : params[3],
                'dataset' : params[4], 'permute' : params[5],
                'pad' : params[6], 'orientation' : params[7]
                }

        json_dump = json.dumps(hyper_dict)
        # open file for writing, "w"
        f = open("../hyperparameter-configurations/hypers.{0}".format(i),"w")

        # write json object to file
        f.write(json_dump)

        # close file
        f.close()

"""
# python dictionary with key value pairs
hyper_dict = {'architecture' : 'exponential', 'learning_rate' : 0.001,
        'dimension' : 128, 'epochs' : 2,
        'dataset' : 'mnist', 'permute' : True,
        'pad' : 0, 'orientation' : None
        }

# create json object from dictionary
json_dump = json.dumps(hyper_dict)

# open file for writing, "w"
f = open("configuration_exponential.json","w")

# write json object to file
f.write(json_dump)

# close file
f.close()

# read file

read_f = open("configuration_anti.json")
hypers = json.load(read_f)
for key in hypers.keys():
        print('key:\t', key)
        print('value:\t', hypers[key], '\n')
"""
