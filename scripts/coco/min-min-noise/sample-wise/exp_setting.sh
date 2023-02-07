#!/bin/bash
#SBATCH --partition=csedu 
# Exp Setting
export config_path=/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/configs/coco/
export dataset_path=/ceph/csedu-scratch/project/mavrikis/mcoco/
export dataset_type=MCOCO
export poison_dataset_type=PoisonMCOCO
export attack_type=min-min
export perturb_type=samplewise
export base_version=resnet18
export epsilon=32
export step_size=3.2
export num_steps=20
export train_step=100
export universal_stop_error=0.1
export universal_train_target='train_dataset'
export exp_args=${dataset_type}-eps=${epsilon}-se=${universal_stop_error}-base_version=${base_version}-number-of-objects-low
export exp_path=experiments/coco/${attack_type}_${perturb_type}/${exp_args}
export scripts_path=scripts/coco/${attack_type}-noise/${perturb_type}
