import random
import numpy as np
import pickle
import copy
import os


class ReplayMemory:
    def __init__(self, capacity, interaction_length):
        self.capacity = capacity
        self.buffer = []
        self.interaction_length = interaction_length
        self.interaction = [None] * self.interaction_length
        self.timestep = 0
        self.position = 0

    # Save the timestep data in an array
    def push_timestep(self, state, action, reward, next_state, done):
        self.interaction[self.timestep] = (state, action, reward, next_state, done)
        self.timestep = (self.timestep + 1) % self.interaction_length

    # Save the interaction array in memory buffer
    def push_interaction(self):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = copy.deepcopy(self.interaction)
        self.position = (self.position + 1) % self.capacity

    # Sample batch of consecutive interactions
    def sample(self, batch_size):
        tau1, tau2, tau3, tau4, tau5 = [], [], [], [], []
        for idx in range(batch_size):
            interaction_idx = random.randint(0, len(self.buffer) - 5)
            tau1.append(self.buffer[interaction_idx + 0])
            tau2.append(self.buffer[interaction_idx + 1])
            tau3.append(self.buffer[interaction_idx + 2])
            tau4.append(self.buffer[interaction_idx + 3])
            tau5.append(self.buffer[interaction_idx + 4])
        return tau1, tau2, tau3, tau4, tau5

    def get_steps(self, interaction_idx):
        return self.buffer[interaction_idx]

    # Get number of interactions
    def __len__(self):
        return len(self.buffer)

    # Save the memory buffer
    def save_buffer(self, name):
        print('[*] Saving buffer as models/rili/buffer_{}.pkl'.format(name))
        if not os.path.exists('models/rili/'):
            os.makedirs('models/rili/')
        with open("models/rili/buffer_{}.pkl".format(name), 'wb') as f:
            pickle.dump(self.buffer, f, protocol=2)

    # Load the memory buffer
    def load_buffer(self, name):
        print('[*] Loading buffer from models/rili/buffer_{}.pkl'.format(name))
        with open("models/rili/buffer_{}.pkl".format(name), "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
