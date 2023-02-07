#!/bin/bash
#SBATCH --partition=csedu -w cn47
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --cpus-per-task=12
#SBATCH --output=my-exp-%J-pertub.out
#SBATCH --error=my-exp-%J-pertub.err
#SBATCH --time=6:00:00
#SBATCH --mail-user=cmavrikis
#SBATCH --mail-type=BEGIN,END,FAIL
# Load Exp Settings
source exp_setting.sh


# Remove previous files
echo $exp_path

#lol
# Search Universal Perturbation and build datasets  25879 or 25337
# for max entropy 25848
# for min entropy 25177
# for 100 random 25427
cd /ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/
pwd
rm -rf $exp_name
python3 perturbation.py --config_path             $config_path       \
                        --exp_name                $exp_path          \
                        --version                 $base_version      \
                        --train_data_path         $dataset_path      \
                        --train_data_type         $dataset_type      \
                        --test_data_path          $dataset_path      \
                        --test_data_type          $dataset_type      \
                        --noise_shape             25427 3 224 224    \
                        --epsilon                 $epsilon           \
                        --num_steps               $num_steps         \
                        --step_size               $step_size         \
                        --attack_type             $attack_type       \
                        --perturb_type            $perturb_type      \
                        --train_step              $train_step        \
                        --train_batch_size        32                 \
                        --eval_batch_size         32                 \
                        --universal_train_target  $universal_train_target\
                        --universal_stop_error    $universal_stop_error\
			--universal_train_portion 0.2 \
                        --use_subset               
