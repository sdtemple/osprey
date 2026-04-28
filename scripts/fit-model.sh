#!/bin/bash
model=$1
table=$2
duration=$3
papermill \
    ../notebooks/fit-$model.ipynb \
    ../notebooks/fit-$model-ran.ipynb \
    -p table_path $table \
    -p output_folder ../results/$model \
    -p num_augments 3 \
    -p num_epochs 20 \
    -p batch_size 256 \
    -p num_workers 8 \
    -p duration $duration \
    -p model_path model.safetensors \
    -p base_folder /nfs/turbo/umor-sethtem/acoustics-data