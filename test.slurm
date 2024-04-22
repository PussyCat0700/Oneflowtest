#!/bin/bash

#SBATCH --job-name=oneflow_test
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=luckywang@sjtu.edu.cn
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err

export http_proxy=http://192.168.102.20:7890
export https_proxy=http://192.168.102.20:7890
export all_proxy=socks://192.168.102.20:7890

set -e
python main.py --run_name resnet50test --model ResNet50 --stage 0 --wandb
python main.py --run_name resnet50test --model ResNet50 --stage 1 --wandb
python main.py --run_name resnet50test --model ResNet50 --stage 2 --wandb
