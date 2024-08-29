#!/bin/bash

# To run your code on a specific GPU x: add "CUDA_VISIBLE_DEVICES=x" before ./run.sh ...

log_dir=Logs

if [ ! -d "$log_dir" ]
then
	mkdir $log_dir
fi

timestamp=`date "+%d.%m.%Y-%H:%M:%S.%3N"`
log_file="$log_dir/$timestamp.log"

if [[ $# -ge 1 ]]
then
	exec=$@
else
	exec=main.py
fi

install_dir=$CONDA_PREFIX/bin

source activate env_pytorch

nohup $install_dir/python -u $exec >> $log_file 2>&1 &

printf "executing $exec - process ID: $!\n" | tee $log_file

