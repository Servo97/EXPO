#!/bin/bash


XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=pen-binary-v0 \
                                --seed=3 \
                                --utd_ratio=20 \
                                --start_training 0 \
                                --max_steps 1000000 \
                                --batch_size=256 \
                                --expo=True \
                                --config=configs/expo_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.edit_action_scale=0.7 \
                                --config.actor_drop=0.1 \
                                --project_name=expo_adroit
