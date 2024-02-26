#!/bin/bash -x
#COBALT -t <time>
#COBALT -n <nodes>
#COBALT -q <queue>
#COBALT -A <project>
#COBALT -O <output>

# --- Prepare Python ---

CONDA_DIR=<path to conda>

# Name of conda environment
export CONDA_ENV_NAME=<environment>

# Activate conda environment
export PYTHONNOUSERSITE=1
source $CONDA_DIR/activate $CONDA_ENV_NAME

# --- Prepare libEnsemble ---

# Name of calling script
export EXE=calling_script.py

# Communication Method
export COMMS='--comms local'

# Number of workers.
export NWORKERS='--nworkers <# workers>'

# Required for killing tasks from workers on Theta
export PMI_NO_FORK=1

# Unload Theta modules that may interfere with task monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

python $EXE $COMMS $NWORKERS > out.txt 2>&1
