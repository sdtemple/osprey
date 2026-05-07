#!/bin/bash
table_path=$1
base_folder=$2
save_folder=$3
n_mels=$4
n_fft=2048
hop_length=512
fmin=0
fmax=16000
papermill \
    ../notebooks/precompute.ipynb \
    ../notebooks/precompute-ran.ipynb \
    -p table_path $table_path \
    -p base_folder $base_folder \
    -p save_folder $save_folder \
    -p n_mels $n_mels \
    -p fmin $fmin \
    -p fmax $fmax \
    -p hop_length $hop_length \
    -p n_fft $n_fft