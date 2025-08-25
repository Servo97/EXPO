#!/bin/bash


XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=antmaze-large-play-v2 \
                                --seed=3 \
                                --utd_ratio=20 \
                                --start_training 5000 \
                                --max_steps 300000 \
                                --expo=True \
                                --config=configs/drlpd_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --config.num_min_qs=1 \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.r_action_scale=0.05 \
                                --project_name=expo_antmaze
