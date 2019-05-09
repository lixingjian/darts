#!/bin/bash

i="$1"
cmd="$2"
sh /home/work/lixingjian/low_level_tasks/kill.sh
sleep 1
source activate pytorch1.0
python -u $cmd
