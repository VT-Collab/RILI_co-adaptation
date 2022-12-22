import argparse
import gym
import gym_rili
import numpy as np
from algos.rili import RILI
from replay_memory import ReplayMemory


parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="rili-circle-v0")
parser.add_argument('--resume', default="None")
parser.add_argument('--change_partner', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save_name')
parser.add_argument('--start_eps', type=int, default=100)
parser.add_argument('--num_eps', type=int, default=30000)
args = parser.parse_args()


# Environment
env = gym.make(args.env_name)
env.set_params(change_partner=args.change_partner)

# Agent
agent = RILI(env.action_space, env.observation_space.shape[0], env._max_episode_steps)

# Memory
memory = ReplayMemory(capacity=args.num_eps, interaction_length=env._max_episode_steps)

# Resume Training
if args.resume != "None":
    agent.load_model(args.resume)
    memory.load_buffer(args.resume)
    args.start_eps = 0

z_prev = np.zeros(10)
z = np.zeros(10)
# Main loop
for i_episode in range(1, args.num_eps+1):

    if len(memory) > 4:
        z = agent.predict_latent(
            memory.get_steps(memory.position - 4),
            memory.get_steps(memory.position - 3),
            memory.get_steps(memory.position - 2),
            memory.get_steps(memory.position - 1))

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:

        if i_episode < args.start_eps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, z)

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        episode_reward += reward

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push_timestep(state, action, reward, next_state, mask)
        state = next_state

    z_prev = np.copy(z)

    memory.push_interaction()
    print("Episode: {}, partner: {}, reward: {}".format(i_episode, env.partner, round(episode_reward, 2)))

