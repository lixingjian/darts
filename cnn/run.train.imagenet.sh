#!/bin/bash

savedir="./"
datadir="/ssd1/image-net"
if [ `hostname` == "svail-1" ]; then
    savedir="/mnt/scratch/lixingjian/darts_v3"
    datadir="/mnt/data/lixingjian//benchmark/imagenet"
fi
mkdir -p log
i=0
r=0
for order in ""; do
for residual_wei in 1 2; do
for lr in 0.05 0.01; do 
    r=`expr $r + 1`
    for ep in 50 125 250; do
        tag="imagenet.train.lr$lr.$i"
        param="train_imagenet.py $order --learning_rate=$lr --residual_wei=$residual_wei $shrink_channel --data $datadir --batch_size=96 --image_size=224 --epochs=$ep --auxiliary --parallel --arch=DARTS_R$r --lr_strategy=cos --bn_no_wd --bias_no_wd --save=$savedir/ckpt.$tag"
        loop=`expr $ep / 2`
        nohup bash run.one.sh $tag "0,1,2,3,4,5,6,7" "$param" $loop 2>&1 >log/log.$tag &
        i=`expr $i + 1`
        sleep 1
    done
done
done
done
