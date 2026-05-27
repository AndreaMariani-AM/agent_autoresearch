#!/bin/bash

module load mpi
module load cuda11.7
module load cudnn8.5-cuda11.7

eval "$(mamba shell hook --shell bash)"
mamba activate /group/glastonbury/conda_envs/lazyslide.v0.9.3

# Set config files
config_file="configs/train_config.yaml"


# Run the Python script for this fold
python ../scripts/train_discriminatorMIL.py \
--config_file $config_file \
--fold 0 \
--input_dim 2560 \
--hidden_dim 1280 \
--output_dim 256 \
--max_epochs 15 \
--lr 1e-4 \
--accumulate_grad_batches 4 \
--weight_decay 1e-3 \
--comment "mammoth_MIL_all_slides" \
--use_discriminator False