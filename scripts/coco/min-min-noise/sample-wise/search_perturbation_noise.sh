#!/bin/bash
#SBATCH --partition=csedu -w cn47
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
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
# for max entropy 34782 for unique 16988
# for min entropy 49208/35501 for unique 28410 for entropy 7687
# for 100 random 33276  for unique 71868/ 35001
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
                        --noise_shape             34782 3 224 224    \
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
