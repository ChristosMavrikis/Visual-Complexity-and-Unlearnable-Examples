#!/bin/bash
#SBATCH --partition=csedu -w cn48
#SBATCH --cpus-per-task=1
#SBATCH --error=imagenet-min-224x224-eq.err
#SBATCH --output=imagenet-min-224x224-eq.out
#SBATCH --mail-user=cmavrikis
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=24:00:00
python3 std_mean.py
