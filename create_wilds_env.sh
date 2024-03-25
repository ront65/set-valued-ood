#!/bin/bash

set -e

# Log file location
LOG_FILE="LOGS_setup_miniconda_wilds.log"
> "$LOG_FILE"

# Function to log messages
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}


# Create a new conda environment with Python 3.10 and activate it
conda create -y -n wilds python=3.10
log "Conda environment 'wilds' created"
source activate wilds
log "Activated conda environment 'wilds'"


pip install wilds
log "installed wilds"

pip install wandb
log "installed wandb"

pip install jupyter
log "installed jupyter"

pip install -e .
log "installed repo"

pip install seaborn
log "installed seaborn"

pip install transformers
log "installed transformers"


