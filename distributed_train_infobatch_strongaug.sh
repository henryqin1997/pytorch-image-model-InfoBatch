#!/bin/bash
NUM_PROC=$1
shift
torchrun --rdzv-endpoint=localhost:13344 --nproc_per_node=$NUM_PROC train_infobatch_strongaug.py "$@"