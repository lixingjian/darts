#!/bin/bash

savedir="./"
if [ `hostname` == "svail-1" ]; then
    savedir="/mnt/scratch/lixingjian/darts_v3"
fi
mkdir -p log
i=0
for order in ""; do
for residual_wei in 1 2; do 
for shrink_channel in "" "--shrink_channel"; do
    tag="cifar.search.$i"
    param="train_search.py $order --residual_wei=$residual_wei $shrink_channel --save=$savedir/ckpt.$tag"
    nohup bash run.one.sh $tag $gpuid "$param" 2>&1 >log/log.$tag &
    i=`expr $i + 1`
done
done
done
