#!/bin/bash
group=$1
table=$2
model=$3
num_workers=$4
bash fit-model.sh \
    ../results/$group \
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
