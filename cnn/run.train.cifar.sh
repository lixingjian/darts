#!/bin/bash

savedir="./"
if [ `hostname` == "svail-1" ]; then
    savedir="/mnt/scratch/lixingjian/darts_v3"
fi
mkdir -p log
i=0
r=0
for residual_wei in 1 2 ; do 
for shrink_channel in "" ""; do
#for shrink_channel in "" "--shrink_channel"; do
    r=`expr $r + 1`
    for ep in 300 600; do
        tag="cifar.train.$i"
        param="train.py --batch_size=64 --epochs=$ep --residual_wei=$residual_wei $shrink_channel --arch=DARTS_R$r --auxiliary --cutout --save=$savedir/ckpt.$tag"
        nohup bash run.one.sh $tag $i "$param" 10 2>&1 >log/log.$tag &
        i=`expr $i + 1`
    done
done
done
