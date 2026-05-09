#!/bin/bash
output=$1
table=$2
base_folder=/nfs/turbo/umor-sethtem/log-spectrogram-128mel-2048fft-512hop-5dur
model=$3
freeze_inner=$4
head_only=$5
loss_name=$6
conduct_cv=$7
num_augments=$8
num_epochs=$9
batch_size=${10}
num_workers=${11}
mel_time_size=313
alpha=${12}
optimizer_name=${13}
use_cosine_annealing=${14}
lr=${15}
mkdir -p "${output}/${model}"
papermill \
    ../notebooks/fit-spec.ipynb \
    $output/$model/fit-spec-ran.ipynb \
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
    -p label_smoothing_alpha $alpha \
    -p freeze_inner $freeze_inner \
    -p head_only $head_only \
    -p conduct_cv $conduct_cv \
    -p optimizer_name $optimizer_name \
    -p use_cosine_annealing $use_cosine_annealing \
    -p lr $lr \
    -k python3