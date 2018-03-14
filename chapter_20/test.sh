#!/bin/sh

# DQN
python main.py --network_header_type=mlp --network_output_type=normal --observation_dims='[64]' --env_name=FrozenLake8x8-v0 --t_learn_start=0.1 --learning_rate_decay_step=0.1 --history_length=1 --n_action_repeat=1 --t_ep_end=50 --display=True --use_gpu=False

# dueling DQN
python main.py --network_header_type=mlp --network_output_type=dueling --observation_dims='[64]' --env_name=FrozenLake8x8-v0 --t_learn_start=0.1 --learning_rate_decay_step=0.1 --history_length=1 --n_action_repeat=1 --t_ep_end=50 --display=True --use_gpu=False

# DDQN
python main.py --network_header_type=mlp --network_output_type=normal --double_q=True --observation_dims='[64]' --env_name=FrozenLake8x8-v0 --t_learn_start=0.1 --learning_rate_decay_step=0.1 --history_length=1 --n_action_repeat=1 --t_ep_end=50 --display=True --use_gpu=False

# Dueling DDQN
python main.py --network_header_type=mlp --network_output_type=dueling --double_q=True --observation_dims='[64]' --env_name=FrozenLake8x8-v0 --t_learn_start=0.1 --learning_rate_decay_step=0.1 --history_length=1 --n_action_repeat=1 --t_ep_end=50 --display=True --use_gpu=False
