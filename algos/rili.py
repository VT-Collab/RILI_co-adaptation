import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from utils import soft_update, hard_update
from algos.model_sac import QNetwork, GaussianPolicy
from algos.model_rili import RILI_Autoencoder


class RILI(object):
    def __init__(self, action_space, state_dim, timestep):

        # hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 0.0003
        self.hidden_size = 256
        self.target_update_interval = 1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]
        self.timestep = timestep

        # Autoencoder
        self.autoencoder = RILI_Autoencoder(state_dim=state_dim, reward_dim=1, latent_dim=10, hidden_dim=64, timesteps=timestep).to(self.device)
        self.ae_optim = Adam(self.autoencoder.parameters(), lr=0.001)

        # Critic
        self.critic = QNetwork(num_inputs=30+state_dim, num_actions=self.action_dim, hidden_dim=self.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(num_inputs=30+state_dim, num_actions=self.action_dim, hidden_dim=self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Actor
        self.policy = GaussianPolicy(num_inputs=30+state_dim, num_actions=self.action_dim, hidden_dim=self.hidden_size, action_space=action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.updates = 0


    def predict_latent(self, interaction1, interaction2, interaction3, interaction4):
        states1, _, rewards1, _, _ = map(np.stack, zip(*interaction1))
        states2, _, rewards2, _, _ = map(np.stack, zip(*interaction2))
        states3, _, rewards3, _, _ = map(np.stack, zip(*interaction3))
        states4, _, rewards4, _, _ = map(np.stack, zip(*interaction4))
        states1 = torch.FloatTensor(states1).to(self.device)
        rewards1 = torch.FloatTensor(rewards1).to(self.device)
        states2 = torch.FloatTensor(states2).to(self.device)
        rewards2 = torch.FloatTensor(rewards2).to(self.device)
        states3 = torch.FloatTensor(states3).to(self.device)
        rewards3 = torch.FloatTensor(rewards3).to(self.device)
        states4 = torch.FloatTensor(states4).to(self.device)
        rewards4 = torch.FloatTensor(rewards4).to(self.device)
        tau1 = torch.cat((states1, rewards1.unsqueeze(1)), 1).flatten()
        tau2 = torch.cat((states2, rewards2.unsqueeze(1)), 1).flatten()
        tau3 = torch.cat((states3, rewards3.unsqueeze(1)), 1).flatten()
        tau4 = torch.cat((states4, rewards4.unsqueeze(1)), 1).flatten()
        z, z_mean, z_std, p, p_mean, p_std = self.autoencoder.encoder(tau1, tau2, tau3, tau4)
        return torch.cat((z, z_std, p), -1).detach().cpu().numpy()


    def predict_latent_gaussian(self, interaction1, interaction2, interaction3, interaction4):
        states1, _, rewards1, _, _ = map(np.stack, zip(*interaction1))
        states2, _, rewards2, _, _ = map(np.stack, zip(*interaction2))
        states3, _, rewards3, _, _ = map(np.stack, zip(*interaction3))
        states4, _, rewards4, _, _ = map(np.stack, zip(*interaction4))
        states1 = torch.FloatTensor(states1).to(self.device)
        rewards1 = torch.FloatTensor(rewards1).to(self.device)
        states2 = torch.FloatTensor(states2).to(self.device)
        rewards2 = torch.FloatTensor(rewards2).to(self.device)
        states3 = torch.FloatTensor(states3).to(self.device)
        rewards3 = torch.FloatTensor(rewards3).to(self.device)
        states4 = torch.FloatTensor(states4).to(self.device)
        rewards4 = torch.FloatTensor(rewards4).to(self.device)
        tau1 = torch.cat((states1, rewards1.unsqueeze(1)), 1).flatten()
        tau2 = torch.cat((states2, rewards2.unsqueeze(1)), 1).flatten()
        tau3 = torch.cat((states3, rewards3.unsqueeze(1)), 1).flatten()
        tau4 = torch.cat((states4, rewards4.unsqueeze(1)), 1).flatten()
        z, z_mean, z_std, p, p_mean, p_std = self.autoencoder.encoder(tau1, tau2, tau3, tau4)
        return torch.cat((z, z_std, p), -1).detach().cpu().numpy(), \
               torch.cat((z_mean, p_mean), -1).detach().cpu().numpy(), \
               torch.cat((z_std, p_std), -1).detach().cpu().numpy()


    def select_action(self, state, z):
            state = torch.FloatTensor(state).to(self.device)
            z = torch.FloatTensor(z).to(self.device)
            context = torch.cat((state, z), 0).unsqueeze(0)
            action, _, _ = self.policy.sample(context)
            return action.detach().cpu().numpy()[0]


    def update_parameters(self, memory, batch_size):

        # Sample a batch from memory
        tau1, tau2, tau3, tau4, tau5 = memory.sample(batch_size=batch_size)

        # States and rewards for first interaction
        states1 = [None] * batch_size
        rewards1 = [None] * batch_size
        for idx, item in enumerate(tau1):
            states1[idx], _, rewards1[idx], _, _ = map(np.stack, zip(*item))

        # States and rewards for second interaction
        states2 = [None] * batch_size
        rewards2 = [None] * batch_size
        for idx, item in enumerate(tau2):
            states2[idx], _, rewards2[idx], _, _ = map(np.stack, zip(*item))

        # States and rewards for third interaction
        states3 = [None] * batch_size
        rewards3 = [None] * batch_size
        for idx, item in enumerate(tau3):
            states3[idx], _, rewards3[idx], _, _ = map(np.stack, zip(*item))

        # States and rewards for fourth interaction
        states4 = [None] * batch_size
        rewards4 = [None] * batch_size
        for idx, item in enumerate(tau4):
            states4[idx], _, rewards4[idx], _, _ = map(np.stack, zip(*item))

        # States and rewards for fifth interaction
        states5 = [None] * batch_size
        actions5 = [None] * batch_size
        rewards5 = [None] * batch_size
        next_states5 = [None] * batch_size
        dones5 = [None] * batch_size
        for idx, item in enumerate(tau5):
            states5[idx], actions5[idx], rewards5[idx], next_states5[idx], dones5[idx] = map(np.stack, zip(*item))

        states1 = torch.FloatTensor(np.array(states1)).to(self.device)
        rewards1 = torch.FloatTensor(np.array(rewards1)).to(self.device)
        states2 = torch.FloatTensor(np.array(states2)).to(self.device)
        rewards2 = torch.FloatTensor(np.array(rewards2)).to(self.device)
        states3 = torch.FloatTensor(np.array(states3)).to(self.device)
        rewards3 = torch.FloatTensor(np.array(rewards3)).to(self.device)
        states4 = torch.FloatTensor(np.array(states4)).to(self.device)
        rewards4 = torch.FloatTensor(np.array(rewards4)).to(self.device)
        states5 = torch.FloatTensor(np.array(states5)).to(self.device)
        actions5 = torch.FloatTensor(np.array(actions5)).to(self.device)
        rewards5 = torch.FloatTensor(np.array(rewards5)).to(self.device)
        next_states5 = torch.FloatTensor(np.array(next_states5)).to(self.device)
        dones5 = torch.FloatTensor(np.array(dones5)).to(self.device)

        tau1 = torch.cat((states1, rewards1.unsqueeze(2)), 2).flatten(start_dim=1)
        tau2 = torch.cat((states2, rewards2.unsqueeze(2)), 2).flatten(start_dim=1)
        tau3 = torch.cat((states3, rewards3.unsqueeze(2)), 2).flatten(start_dim=1)
        tau4 = torch.cat((states4, rewards4.unsqueeze(2)), 2).flatten(start_dim=1)
        tau5 = torch.cat((states5, rewards5.unsqueeze(2)), 2).flatten(start_dim=1)

        # train the autoencoder
        ae_loss, curr_loss, next_loss, kl_loss = self.autoencoder(tau1, tau2, tau3, tau4, states1, rewards1, states2, rewards2, states3, rewards3, states4, rewards4, states5, rewards5)

        self.ae_optim.zero_grad()
        ae_loss.backward()
        self.ae_optim.step()

        # get latent's for states and next states
        z_curr, _, z_curr_std, p_curr, _, _ = self.autoencoder.encoder(tau1, tau2, tau3, tau4)
        z_plus, _, z_plus_std, p_plus, _, _ = self.autoencoder.encoder(tau2, tau3, tau4, tau5)
        latent_curr = torch.cat((z_curr, z_curr_std, p_curr), -1).detach()
        latent_plus = torch.cat((z_plus, z_plus_std, p_plus), -1).detach()
        latent_curr = torch.repeat_interleave(latent_curr, self.timestep, dim=0)
        latent_next = torch.clone(latent_curr)
        for idx, item in enumerate(latent_plus):
            latent_next[self.timestep*(idx+1)-1] = item

        state_batch = torch.reshape(states5, (-1, self.state_dim))
        state_batch = torch.cat((state_batch, latent_curr), 1)
        next_state_batch = torch.reshape(next_states5, (-1, self.state_dim))
        next_state_batch = torch.cat((next_state_batch, latent_next), 1)
        action_batch = torch.reshape(actions5, (-1, self.action_dim))
        reward_batch = torch.reshape(rewards5, (-1, 1))
        mask_batch = torch.reshape(dones5, (-1, 1))

        # train critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # train actor
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        self.updates += 1

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), ae_loss.item(), curr_loss.item(), next_loss.item(), kl_loss.item()


    def save_model(self, name):

        print('[*] Saving RILI as models/rili/{}.pt'.format(name))
        if not os.path.exists('models/rili/'):
            os.makedirs('models/rili/')

        checkpoint = {
            'updates': self.updates,
            'autoencoder': self.autoencoder.state_dict(),
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'ae_optim': self.ae_optim.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict()
        }
        torch.save(checkpoint, "models/rili/{}.pt".format(name))


    def load_model(self, name):

        print('[*] Loading RILI from models/rili/{}.pt'.format(name))

        checkpoint = torch.load("models/rili/{}.pt".format(name), map_location=self.device)
        self.updates = checkpoint['updates']
        self.autoencoder.load_state_dict(checkpoint['autoencoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])
        self.ae_optim.load_state_dict(checkpoint['ae_optim'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
