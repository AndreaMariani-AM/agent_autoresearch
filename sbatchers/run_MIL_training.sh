#!/bin/bash
#SBATCH --job-name="MIL_training"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.mariani@fht.org
#SBATCH --partition=gpuq
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH --output=/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/experiments/training/hierarchical_embeddings_128/training_fold_%a.log

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
--fold ${SLURM_ARRAY_TASK_ID} \
--model_type "hierarchical" \
--input_dim 2560 \
--pooling "sum" \
--frozen_backbone "Virchow2" \
--scale_drop_p 0.15 \
--max_epochs 500 \
--accumulate_grad_batches 12 \
--cell_warmup_start 20 \
--loss "focal" \
--lr 1e-4 \
--weight_decay 1e-3 \
--top_k 384 \
--patient_level \
--comment "focalLoss_patient_aware"
