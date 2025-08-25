#!/bin/bash


XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=antmaze-large-play-v2 \
                                --seed=3 \
                                --utd_ratio=20 \
                                --pretrain_edit=False \
                                --pretrain_q=False \
                                --pretrain_steps=500000 \
                                --offline_eval_interval=50000 \
                                --start_training 0 \
                                --max_steps 300000 \
                                --expo=True \
                                --config=configs/drlpd_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --config.num_min_qs=1 \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.edit_action_scale=0.05 \
                                --project_name=expo_antmaze
