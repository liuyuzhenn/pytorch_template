#!/bin/bash

GPUS=$1

torchrun --nproc_per_node=$GPUS train.py ${@:2}
