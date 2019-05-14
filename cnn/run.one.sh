#!/bin/bash

tag="$1"
gpus="$2"
cmd="$3"
loop="$4"
ngpu=`echo $gpus | awk -F',' '{print NF}'`

echo $tag,$ngpu,$loop,$cmd

export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
source activate pytorch1.0
#source activate pt1.0_py3.6
if [ `hostname` == "svail-1" ]; then
    for((j=0;j<$loop;j++)); do
        if [ `sinfo |grep idle |sed 's/\s\+/ /g' |grep "1080Ti up" |wc -l` -eq 1 ]; then
          part="--partition=1080Ti --cpus-per-task=$((2 * $ngpu))"
        elif [ `sinfo |grep idle |sed 's/\s\+/ /g' |grep "TitanXx8 up" |wc -l` -eq 1 ]; then
          part="--partition=TitanXx8,M40x8 --cpus-per-task=$((5 * $ngpu))"
        else
          part="--partition=TitanXx8,M40x8 --cpus-per-task=$((5 * $ngpu))"
        fi
        srun --job-name=$tag $part --gres=gpu:$ngpu -n1 python -u $cmd
        #srun --job-name=$tag $part --gres=gpu:$ngpu -n1 bash run.sh "$cmd"
    done
else
    sh /home/work/lixingjian/low_level_tasks/kill.sh
    CUDA_VISIBLE_DEVICES=$gpus python -u $cmd
fi
