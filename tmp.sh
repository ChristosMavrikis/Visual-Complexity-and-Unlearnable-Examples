#!/bin/bash
#SBATCH --partition=csedu 
#SBATCH --cpus-per-task=1
#SBATCH --error=imagenet-low-gray.err
#SBATCH --output=imagenet-low-gray.out
#SBATCH --mail-user=cmavrikis
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=24:00:00
python3 display_imagenet.py
#python3 entropy_min.py
#python3 entropy_min.py
#python3 targets.py
#python3 coco-label.py
#python3 plot_histogram_imagenet.py
