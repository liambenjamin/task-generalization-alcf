# Interrogating Task Generalization
Supporting codes for the paper *Interrogating Task Generalization of Model Behaviors on Benchmark Tasks*.

## Repository Organization
Code is organized in two structures corresponding to the top level directories: *quick-deployment* and *theta-deployment*. Codes in *quick-deployment* allow for quickly configuring an experimental training attempt and initializing training. The codes in *theta-deployment* correspond to the routine used to deploy batches of training attempts across Argonne National Lab's supercomputer, Theta. Additional information on Theta can be found [here](https://www.alcf.anl.gov/systems/theta). As of January 1, 2024, Theta has been retired and is no longer in use.


### Quick Deployment File Contents

We provide a jupyter notebook `sample_train_task.ipynb` that walks through loading a recurrent architecture and particular experimental task. The notebook relies on the accompanying python files:

	1. `architectures.py` : implementations of various recurrent architectures
	
	2. `network.py` : code for generating recurrent neural network from keras model API
	
	3. `load_data.py` : code for loading task data (relies on `tensorflow.keras.datasets`)
	


### Accessing Data

The datasets used for training need to be downloaded once prior to running training scripts. Once the datasets are downloaded, internet connection is no longer needed. To download the datasets open a python session and run:

	> from tensorflow.keras.datasets import mnist, fashion_mnist
	> mnist.load_data()
	> fashion_mnist.load_data()
	> exit()

This will download the datasets to ``~/.keras/datasets`` and make them available for the training script to access at runtime. Executing the training script will load and augment the downloaded datasets according to the arguments stored in ``hypers.<id>``. 


### Theta Deployment File Contents

The experiment deployment across Theta utilzed the python library [libEnsemble](https://libensemble.readthedocs.io/en/main/). Path names specific to the filesystem at Argonne Leadership Computing Facility (ALCF) have been removed from the original codes.
