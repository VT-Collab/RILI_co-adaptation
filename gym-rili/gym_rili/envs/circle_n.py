import numpy as np
import gym
from gym import spaces


# Ego agent localisation
ego_home = np.array([0.0, 0.5])


class Circle_N(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(
            low=-0.2,
            high=+0.2,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(2,),
            dtype=np.float32
        )

        self.radius = 1.0
        self.change_partner = 0.99
        self.step_size = np.random.random() * 2 * np.pi - np.pi
        self.ego = np.copy(ego_home)
        self.other = np.array([self.radius, 0.])
        self.theta = 0.0
        self.partner = 0
        self.timestep = 0


    def set_params(self, change_partner):
        self.change_partner = change_partner


    def _get_obs(self):
        return np.copy(self.ego)


    def polar(self, theta):
        return self.radius * np.array([np.cos(theta), np.sin(theta)])


    def reset(self):
        return self._get_obs()


    def step(self, action):
        self.timestep += 1
        self.ego += action
        reward = -np.linalg.norm(self.other - self.ego) * 100
        done = False
        if self.timestep == 10:
            self.timestep = 0
            # choose a new partner with random step size
            if np.random.random() > self.change_partner:
                self.partner += 1
                self.step_size = np.random.random() * 2 * np.pi - np.pi
            self.theta += self.step_size
            self.ego = np.copy(ego_home)
            self.other = self.polar(self.theta)
        return self._get_obs(), reward, done, {}

