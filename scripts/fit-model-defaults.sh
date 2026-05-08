#!/bin/bash
output=$1
table=$2
model=$3
num_workers=$4
bash fit-model.sh \
    $output \
    $table \
    $model \
    False \
    False \
    bce \
    True \
    10 \
    256 \
    $num_workers \
    0.05 \
    adam \
    False \
    4e-4
