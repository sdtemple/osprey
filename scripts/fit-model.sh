#!/bin/bash
output=$1
table=$2
base_folder=/nfs/turbo/umor-sethtem/log-spectrogram-128mel-2048fft-512hop-5dur
model=$3
loss_name=$4
num_augments=$5
num_epochs=$6
batch_size=256
num_workers=$7
mel_time_size=313
alpha=0.05
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
    -p model_path model.safetensors \
    -p base_folder $base_folder  \
    -p mel_time_size $mel_time_size \
    -p label_smoothing_alpha $alpha