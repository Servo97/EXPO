# Expressive Policy Optimization (EXPO)

![alt text](plots/offline_to_online.png "Title")

Code for the paper "EXPO: Stable Reinforcement Learning with Expressive Policies", available [here](https://arxiv.org/abs/2507.07986)

This code is built on top of the [jaxrl](https://github.com/ikostrikov/jaxrl) framework and the [RLPD](https://github.com/ikostrikov/rlpd
) repository. 

# Installation

```bash
conda env create -f environment.yml
conda activate expo
conda install patchelf  # If you use conda.
pip install -r requirements.txt
conda deactivate
conda activate expo
```

# Experiments
Example scripts are provided in the scripts directory.
## D4RL Antmaze
train_finetuning.py is used for D4RL Antmaze and Adroit experiments. 
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=antmaze-large-play-v2 \
                                --seed=3 \
                                --utd_ratio=20 \
                                --start_training 5000 \
                                --max_steps 300000 \
                                --expo=True \
                                --config=configs/expo_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --config.num_min_qs=1 \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.edit_action_scale=0.05 \
                                --project_name=expo
```

## Adroit Binary

First, download and unzip `.npy` files into `~/.datasets/awac-data/` from [here](https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y/view).

Make sure you have `mjrl` installed:
```bash
git clone https://github.com/aravindr93/mjrl
cd mjrl
pip install -e .
```

Then, recursively clone `mj_envs` from this fork:
```bash
git clone --recursive https://github.com/philipjball/mj_envs.git
```

Then sync the submodules (add the `--init` flag if you didn't recursively clone):
```bash
$ cd mj_envs  
$ git submodule update --remote
```

Finally:
```bash
$ pip install -e .
```
May need to remove mujoco dependency in setup.py of mj_env

## Robomimic

train_robo.py is used fro Robomimic and MimicGen environments.

Download the datasets from [here](https://robomimic.github.io/docs/v0.3/datasets/robomimic_v0.1.html) and put it in ./robomimic/datasets/{env_name}/ph for ph and robomimic/datasets/{env_name}/mh for mh. 


```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_robo.py --env_name=square \
                                --seed=3 \
                                --utd_ratio=20 \
                                --dataset_dir='ph' \
                                --start_training 5000 \
                                --max_steps 500000 \
                                --config=configs/expo_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.edit_action_scale=0.05 \
                                --project_name=expo
```


## MimicGen

Install MimicGen by running the following command

```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
```


Download the datasets from [here](https://drive.google.com/file/d/1qemTmLkEkE17dFN6A2BJvVYLJBtLbN67/view?usp=sharing) and put it in ./mimicgen/datasets/{env_name}/. The datasets are subsampled from the original MimicGen datasets. 


Then, run with 

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_robo.py --env_name=threading \
                                --seed=3 \
                                --utd_ratio=20 \
                                --start_training 5000 \
                                --max_steps 500000 \
                                --config=configs/expo_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --config.N=8 \
                                --config.n_edit_samples=8 \
                                --config.edit_action_scale=0.05 \
                                --project_name=expo
```


## Citation


```bash
@misc{dong2025expo,
      title={EXPO: Stable Reinforcement Learning with Expressive Policies}, 
      author={Perry Dong and Qiyang Li and Dorsa Sadigh and Chelsea Finn},
      year={2025},
      eprint={2507.07986},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.07986}, 
}
```

