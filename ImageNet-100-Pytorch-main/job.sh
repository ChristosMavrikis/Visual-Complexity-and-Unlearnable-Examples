#!/bin/bash
#SBATCH --partition=csedu -w cn48
#SBATCH --output=my-exp.out
#SBATCH --err=my-exp.err
python generate_IN100.py \
	--source_folder /scratch/data_share_ChDi/ILSVRC2012/train\
	--target_folder /scratch/data_share_ChDi/Imagenet-100/train\
	--target_class tmp/min100.txt
