import argparse
import datetime
import os.path
import gym
import gym_rili
import numpy as np
from algos.rili import RILI
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="rili-circle-v0")
parser.add_argument('--resume', default="None")
parser.add_argument('--change_partner', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save_name', default='run')
parser.add_argument('--start_eps', type=int, default=300)
parser.add_argument('--num_eps', type=int, default=30000)
args = parser.parse_args()


# Environment
env = gym.make(args.env_name)
env.set_params(change_partner=args.change_partner)

# Agent
agent = RILI(env.action_space, env.observation_space.shape[0], env._max_episode_steps)

# Tensorboard
folder = "runs/" + 'rili' + "/"
writer = SummaryWriter(folder + '{}_{}'.format(args.save_name, datetime.datetime.now().strftime("%m-%d_%H-%M")))

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

        if len(memory) > args.batch_size:
            critic_1_loss, critic_2_loss, policy_loss, ae_loss, curr_loss, next_loss, kl_loss = agent.update_parameters(memory, args.batch_size)
            writer.add_scalar('autoencoder/ae_loss', ae_loss, agent.updates)
            writer.add_scalar('autoencoder/z_curr_loss', curr_loss, agent.updates)
            writer.add_scalar('autoencoder/z_next_loss', next_loss, agent.updates)
            writer.add_scalar('autoencoder/kl_loss', kl_loss, agent.updates)
            writer.add_scalar('SAC/critic_1', critic_1_loss, agent.updates)
            writer.add_scalar('SAC/critic_2', critic_2_loss, agent.updates)
            writer.add_scalar('SAC/policy', policy_loss, agent.updates)

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        episode_reward += reward

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push_timestep(state, action, reward, next_state, mask)
        state = next_state

    z_prev = np.copy(z)

    memory.push_interaction()
    writer.add_scalar('reward/episode_reward', episode_reward, i_episode)
    print("Episode: {}, partner: {}, reward: {}".format(i_episode, env.partner, round(episode_reward, 2)))

    if i_episode % 5000 == 0:
        agent.save_model(args.save_name + '_' + str(i_episode))
        memory.save_buffer(args.save_name + '_' + str(i_episode))

agent.save_model(args.save_name + '_' + str(i_episode))
memory.save_buffer(args.save_name + '_' + str(i_episode))
