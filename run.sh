#!/bin/bash
#SBATCH -J i1k
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=chenxiang.zhang@uni.lu
#SBATCH --account=p200535
#SBATCH --qos=default
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
#SBATCH --output=slurm/%x-%j.out

echo -e "--------------------------------"
echo -e "Start:\t $(date)"
echo -e "JobID:\t ${SLURM_JOBID}"
echo -e "Node:\t ${SLURM_NODELIST}"
echo -e "--------------------------------\n"

eval "$(micromamba shell hook --shell bash)"
micromamba activate imagenet

BS=256  # per GPU (total = 256 * 4 = 1024)
LR=0.001
EP=90
MOM=0.9
OPT=adam
DATA=i1k
WARM_RATIO=0.1

DIR_DATA=/project/home/p200535/data/imagenet1k
DIR_OUTPUT=/project/home/p200535/project/adam-sgd-gap
DIR_OUTPUT=${DIR_OUTPUT}/lr${LR}_bs$((BS*4))_ep${EPOCHS}

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --bs $BS \
    --lr $LR \
    --mom $MOM \
    --aug \
    --opt $OPT \
    --data $DATA \
    --epochs $EP \
    --warm_ratio $WARM_RATIO \
    --dir_data $DIR_DATA \
    --dir_output $DIR_OUTPUT \
    --n_workers 16 \
    --compile
