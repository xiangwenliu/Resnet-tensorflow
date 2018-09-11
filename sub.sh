#!/bin/bash
#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30720M   # memory per CPU core
#SBATCH -J "train"   # job name
./myscript.sh

