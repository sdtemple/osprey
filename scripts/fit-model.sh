#!/bin/bash
output=$1
table=$2
base_folder=$3
model=$4
height=$5
width=$height
loss_name=$6
num_augments=$7
num_epochs=$8
batch_size=$9
num_workers=${10}
mel_time_size=${11}
papermill \
    ../notebooks/fit-spec.ipynb \
    ../notebooks/fit-spec-$model-ran.ipynb \
    -p table_path $table \
    -p output_folder $output/$model \
    -p num_augments $num_augments \
    -p num_epochs $num_epochs \
    -p batch_size $batch_size \
    -p num_workers $num_workers \
    -p loss_name $loss_name \
    -p model_name $model \
    -p height $height \
    -p width $width \
    -p model_path model.safetensors \
    -p base_folder $base_folder \
    -p mel_time_size $mel_time_size