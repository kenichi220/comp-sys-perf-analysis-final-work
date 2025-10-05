#!/bin/bash -l

#SBATCH --job-name=pre-process
#SBATCH --partition=draco
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00

cd $SCRATCH

git clone git@github.com:kenichi220/comp-sys-perf-analysis-final-work.git

cd comp-sys-perf-analysis-final-work/tensorflow

./create-env.sh

source venv/bin/activate

python preprocess_cifar.py