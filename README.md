# alcf-rnn-robustness
Codes supporting RNN project at ALCF.

## File Contents

The model training script is located at ``src/train.py``. This script takes training arguments located at ``src/hyperparameter-configurations/hypers.<id>``. Each argument configuration ``hypers.<id>`` contains a dictionary of arguments that is parsed and fed to the training script with ``sys.argv``. 
