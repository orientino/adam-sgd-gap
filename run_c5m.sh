#!/bin/bash
#SBATCH -J c10
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=chenxiang.zhang@uni.lu
#SBATCH --account=p200535
#SBATCH --qos=default
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm/%x-%j.out

echo -e "--------------------------------"
echo -e "Start:\t $(date)"
echo -e "JobID:\t ${SLURM_JOBID}"
echo -e "Node:\t ${SLURM_NODELIST}"
echo -e "--------------------------------\n"

eval "$(micromamba shell hook --shell bash)"
micromamba activate imagenet

SEED=42
BS=256
EP=1
MOM=0.9
OPT=adam
DATA=c5m
WARM_RATIO=0.1

DIR_DATA=/project/home/p200535/data
DIR_OUTPUT=/project/home/p200535/project/adam-sgd-gap/$DATA

LRS=(1e-4 3e-4 1e-3 3e-3)

for GPU in 0 1 2 3; do
    LR=${LRS[$GPU]}

    NAME=${OPT}_lr${LR}_bs${BS}_mom${MOM}_ep${EP}_seed${SEED}
    DIR_OUTPUT=$DIR_OUTPUT/$NAME

    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --seed $SEED \
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
        --n_workers 8 \
        --log_interval 10 \
        --eval_interval 10 \
        --compile \
        --debug &
done

wait
