#!/bin/bash
#SBATCH --partition=csedu -w cn47
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=my-exp-%J-train.out
#SBATCH --error=my-exp-%J-train.err
#SBATCH --mail-user=cmavrikis
#SBATCH --mail-type=BEGIN,END,FAIL

# Load EXP Setting
source exp_setting.sh


# Training Setting
model_name=resnet18
poison_rate=1.0
exp_name=${exp_path}/poison_train_${poison_rate}
echo $exp_name

# Poison Training
#cd /scratch/data_share_ChDi/Unlearnable-Examples-main/
cd /ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/
rm -rf ${exp_name}/${model_name}
python3 -u main.py    --version                 $model_name                 \
                      --exp_name                $exp_name                   \
                      --config_path             $config_path                \
                      --train_data_path         $dataset_path               \
                      --train_data_type         $poison_dataset_type        \
                      --test_data_path          $dataset_path               \
                      --test_data_type          $dataset_type               \
                      --poison_rate             $poison_rate                \
                      --perturb_type            $perturb_type               \
                      --perturb_tensor_filepath ${exp_path}/perturbation.pt \
                      --train 
		      #--train_portion 0.2
