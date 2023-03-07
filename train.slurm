#!/bin/bash

### SLURM Directives
#SBATCH --job-name="basesmall"
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/person.err
#SBATCH --output=logs/person.out


### Run program
echo "*running node: ${SLURM_NODELIST}"
nvidia-smi
which nvcc

source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

python3 trainer.py\
 --epochs 51 --lr_name 'adam' --lr 0.0001\
 --stepsize_lr 30 --gamma_lr 0.1\
 --weight_decay 0.0001 --batch_size_train 32\
 --batch_size_val 8 --num_workers 8\
 --name 'COCO_SMALL' --seed 12345 --mode 'segm'\
 --vit_type 'small' --augmentation 'ssdlite'\
 --img_resize 224 --version 'baseline/FPNv2_ViT'\
 --cat_list 1\