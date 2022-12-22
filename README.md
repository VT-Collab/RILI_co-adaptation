# Learning Latent Representations to Co-Adapt to Humans

This is a minimal repository for our paper "Learning Latent Representations to Co-Adapt to Humans" [http://arxiv.org/abs/2212.09586]. We include two main things:
- Code for our proposed alogrithm RILI: **R**obustly **I**nfluencing **L**atent **I**ntent
- Four custom gym environments we used during our simulated experiments

## Requirements
- python (v3.9)
- pytorch (v1.14)
- gym (v0.23)
- pybullet (v3.2)
- setuptools (v63.4)

You can install the above packages by running the following command

`
pip install -r requirements.txt
`

Then, install the gym environment:

```
cd gym_rili
pip install -e .
cd ..
```

## Instructions
To train the RILI model with the _Circle_ environment, run the following:

`python3 main.py --env-name rili-circle-v0 --save_name []`

You can train the model in different environments using the `--env-name` argument. It has the following
values:

- `rili-circle-v0`
- `rili-circle-N-v0`
- `rili-driving-v0`
- `rili-robot-v0`

To test the trained models, run the `evaluate.py` script. Pass the correct environment name, and
use the `--resume` argument to pass the name of the saved model that you want to load:

`python3 evaluate.py --resume [model_to_load] --env-name [env_name]`

Other parameters that you can change are:
- `--change_partner`: the stochasticity with which the partner changes (a value between 0 and 1) 
- `--num_eps`: the number of episodes for training
- `--start_eps`: the number of episodes for exploration

We have pre-trained checkpoints saved in the repository for each environment. You can load the desired
model by passing the name in the `--resume` argument:

- For _Circle_ environment:
  - `--resume circle_env_30000`

- For _Circle-N_ environment:
  - `--resume circle_N_env_30000`

- For _Driving_ environment:
  - `--resume driving_env_30000`

- For _Robot_ environment:
  - `--resume robot_env_30000`

