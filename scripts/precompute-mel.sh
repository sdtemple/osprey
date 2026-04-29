#!/bin/bash
table_path=$1
base_folder=$2
save_folder=$3
height=$4
width=$5
papermill \
    ../notebooks/precompute.ipynb \
    ../notebooks/precompute-ran.ipynb \
    -p table_path $table_path \
    -p base_folder $base_folder \
    -p save_folder $save_folder \
    -p height $height \
    -p width $width