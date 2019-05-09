#!/bin/bash

mkdir -p log
i=0
for order in ""; do
for residual_wei in 1 2 ; do 
for shrink_channel in "" "--shrink_channel"; do
    param="train_search.py --gpu=$i $order --residual_wei=$residual_wei $shrink_channel --save=ckpt.$i"
    nohup bash run.one.sh $i "$param" 2>&1 >log/log.$i &
    i=`expr $i + 1`
done
done
done
