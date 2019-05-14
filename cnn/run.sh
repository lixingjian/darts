#!/bin/bash

cmd="$1"

source activate pt1.0_py3.6
#source activate pytorch1.0
/sbin/ifconfig
nvidia-smi
python -u $cmd
