#!/bin/bash

set -e


# Log file location
LOG_FILE="LOGS_setup_miniconda_wilds.log"
> "$LOG_FILE"

# Function to log messages
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

conda create -n fmow numpy=1.21.6 scipy=1.7.3 pandas=1.3.5
log "Conda environment 'fmow' created"
source conda activate fmow
log "Activated conda environment 'fmow'"


pip install wilds
log "installed wilds"

pip install wandb
log "installed wandb"

pip install -e .
log "installed repo"

pip install transformers
log "installed transformers"
