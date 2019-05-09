#!/bin/bash

tag="$1"
gpus="$2"
cmd="$3"
ngpu=`echo $gpus | awk -F',' '{print NF}'`

echo $tag,$ngpu,$cmd

source activate pytorch1.0
if [ `hostname` == "svail-1" ]; then
    part="--partition=TitanXx8,M40x8 --cpus-per-task=$((5 * $ngpu))"
    echo $part
    srun --job-name=$tag $part --gres=gpu:$ngpu -n1 python -u $cmd
else
    sh /home/work/lixingjian/low_level_tasks/kill.sh
    CUDA_VISIBLE_DEVICES=$gpus python -u $cmd
fi
