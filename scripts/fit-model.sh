#!/bin/bash
model=$1
table=$2
num_augments=$3
num_epochs=$4
batch_size=$5
num_workers=$6
base_folder=$7
duration=$8
papermill \
    ../notebooks/fit-$model.ipynb \
    ../notebooks/fit-$model-ran.ipynb \
    -p table_path $table \
    -p output_folder ../results/$model \
    -p num_augments $num_augments \
    -p num_epochs $num_epochs \
    -p batch_size $batch_size \
    -p num_workers $num_workers \
    -p duration $duration \
    -p model_path model.safetensors \
    -p base_folder $base_folder