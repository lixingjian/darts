#!/bin/sh

if [ "$1" == "cifar.train" ]; then
    ps axu |grep "bash run.one.sh $1" |sed 's/\s\+/ /g' |cut -d' ' -f2 |xargs -i kill -9 {}
    bash ~/transfer_learning/killall.sh cifar.tr
elif [ "$1" == "imagenet.train" ]; then
    ps axu |grep "bash run.one.sh $1" |sed 's/\s\+/ /g' |cut -d' ' -f2 |xargs -i kill -9 {}
    bash ~/transfer_learning/killall.sh imagenet
elif [ "$1" == "imagenet.search" ]; then
    ps axu |grep "bash run.one.sh $1" |sed 's/\s\+/ /g' |cut -d' ' -f2 |xargs -i kill -9 {}
    bash ~/transfer_learning/killall.sh imagenet
fi
