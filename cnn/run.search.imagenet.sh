#!/bin/bash

savedir="./"
datadir="/ssd1/image-net"
if [ `hostname` == "svail-1" ]; then
    savedir="/mnt/scratch/lixingjian/darts_v3"
    datadir="/mnt/data/lixingjian//benchmark/imagenet.mini"
fi
mkdir -p log
i=0
for residual_wei in 1 0.5; do 
    tag="imagenet.search.$i"
    param="train_search_imagenet.py --batch_size=64 --input_scale=0.5 --channel_scale=2 --epoch=100 --data $datadir --residual_wei=$residual_wei --save=$savedir/ckpt.$tag"
    nohup bash run.one.sh $tag $i "$param" 10 2>&1 >log/log.$tag &
    i=`expr $i + 1`
done
