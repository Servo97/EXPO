#!/bin/bash

XLA_PYTHON_CLIENT_PREALLOCATE=false python train_robo.py --env_name=square \
                                --seed=3 \
                                --dataset_dir=ph \
                                --utd_ratio=20 \
                                --start_training 5000 \
                                --max_steps 1000000 \
                                --config=configs/expo_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --config.num_min_qs=2 \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.edit_action_scale=0.05 \
                                --project_name=expo_square
