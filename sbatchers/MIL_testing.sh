#!/bin/bash
#SBATCH --job-name="MIL_testing"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.mariani@fht.org
#SBATCH --partition=gpuq
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --output=/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/experiments/testing/training_fold_%a.log

module load mpi
module load cuda11.7
module load cudnn8.5-cuda11.7

eval "$(mamba shell hook --shell bash)"
mamba activate /group/glastonbury/conda_envs/lazyslide.v0.9.3

# Set config files
config_file="../configs/train_config.yaml"

# Run the Python script for this fold
srun python ../scripts/train_MIL.py \
--config_file $config_file \
--fold 0 \
--model_type "hierarchical" \
--pooling "sum" \
--frozen_backbone "Virchow2" \
--top_k 128 \
--scale_drop_p 0.15 \
--aux_loss_lambda 0.3 \
--max_epochs 100 \
--accumulate_grad_batches 32 \
--label_smoothing 0.05 \
--comment "label_smoothing_lastLayerOne" \
--all_embeddings