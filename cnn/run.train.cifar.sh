#!/bin/bash

mkdir -p log
i=0
r=0
for residual_wei in 1 2 ; do 
for shrink_channel in "" "--shrink_channel"; do
    r=`expr $r + 1`
    for ep in 300 600; do
        param="train.py --epochs=$ep --residual_wei=$residual_wei $shrink_channel --gpu=$i --arch=DARTS_R$r --auxiliary --cutout --save=ckpt.train.$i"
        nohup bash run.one.sh $i "$param" 2>&1 >log/log.train.$i &
        i=`expr $i + 1`
    done
done
done
