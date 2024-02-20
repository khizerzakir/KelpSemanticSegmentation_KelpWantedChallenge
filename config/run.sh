#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH --constraint a6000
#SBATCH --constraint m48
#SBATCH --mem 10G
#SBATCH --time 10:00:00
#SBATCH --partition shortrun
#SBATCH --output=kelp_segmentation_%j.out

if [[ ! -z ${SLURM_JOBID+z} ]]; then
    echo "Setting up SLURM environment"
    # Load the Conda environment
    source /share/common/anaconda/etc/profile.d/conda.sh
    conda activate kelp_segmentation
else
    echo "Not a SLURM job"
fi

set -o errexit -o pipefail -o nounset

RAW_DATA="/share/projects/erasmus/KelpMapping/data/raw"
PROCESSED_DATA="/share/projects/erasmus/KelpMapping/data/processed"
OUTPUTS="/share/projects/erasmus/KelpMapping/outputs"
SPLIT_MODE="train_val_test"
BATCH_SIZE=16
MODEL="unet_nested"
LOSS_FUNC="DiceLoss"
LR=0.0001
WEIGHT_DECAY=1e-7
OPTIMIZER="Adam"
NUM_EPOCHS=80

echo "Starting script"
echo $(date)

python main.py \
    --raw_data $RAW_DATA \
    --processed_data $PROCESSED_DATA \
    --outputs $OUTPUTS \
    --split_mode $SPLIT_MODE \
    --batch_size $BATCH_SIZE \
    --use_distance_maps \
    --model $MODEL \
    --loss_func $LOSS_FUNC \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --num_epochs $NUM_EPOCHS

echo $(date)
