# alcf-rnn-robustness
Codes supporting RNN project at ALCF.

## File Contents

The model training script is located at ``src/train.py``. This script takes training arguments located in the ``hyperparameter-configurations`` directory. Each argument configuration is named ``hypers.<id>``, and contains a dictionary of arguments that is parsed and fed to the training script with ``sys.argv``.

As an example, training a single model can be performed from the terminal by executing:

	python train.py hyperparameter-configurations/hypers.1  
