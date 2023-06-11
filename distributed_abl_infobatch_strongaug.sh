#!/bin/bash
NUM_PROC=$1
shift
torchrun --master_port 13344 --nproc_per_node=$NUM_PROC train_infobatch_strongaug_abl.py "$@"