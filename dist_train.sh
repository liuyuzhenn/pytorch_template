#!/bin/bash

GPUS=$1
PORT=${PORT:-29500}

torchrun --master_port $PORT --nproc_per_node=$GPUS train.py ${@:2}