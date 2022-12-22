import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


# Starting Position for Our Car
ego_home = np.array([0.0, 0.0])
other_lanes = np.array([[-1.5, 10.], [-0.5, 10.], [+0.5, 10.], [+1.5, 10.]])


class Driving(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(
            low=-0.2,
            high=+0.2,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(2,),
            dtype=np.float32
        )

        self.car_width = 1.0
        self.change_partner = 0.99
        self.ego = np.copy(ego_home)
        self.other = np.array([0.0, 10.])
        self.partner = 0
        self.prev_lane = 0
        self.other_lane = 0
        self.timestep = 0


    def set_params(self, change_partner):
        self.change_partner = change_partner


    def _get_obs(self):
        return np.copy(self.ego)


    def reset(self):
        return self._get_obs()


    def step(self, action):
        self.timestep += 1

        if self.timestep == 7:
            if self.partner == 0:
                if self.ego[0] > 0.:
                    self.other[0] = 0.5
                else:
                    self.other[0] = self.ego[0]

            elif self.partner == 1:
                if self.ego[0] < 0.:
                    self.other[0] = -0.5
                else:
                    self.other[0] = self.ego[0]

        self.ego += np.array([action[0], 1.0])
        reward = -abs(action[0]) * 10
        done = False

        if self.timestep == 10:
            self.timestep = 0

            # Check for collision
            if abs(self.ego[0] - self.other[0]) < self.car_width:
                reward -= 100

            # choose a new partner from the three options
            if np.random.rand() > self.change_partner:
                self.partner = np.random.choice(range(5))

            if self.partner == 2:
                self.other_lane = (self.other_lane - 1) % 4
                self.other = np.copy(other_lanes[self.other_lane])
            elif self.partner == 3:
                self.other_lane = (self.other_lane + 1) % 4
                self.other = np.copy(other_lanes[self.other_lane])
            elif self.partner == 4:
                self.other[0] = self.prev_lane[0]

            self.prev_lane = np.copy(self.ego)
            self.ego = np.copy(ego_home)
        return self._get_obs(), reward, done, {}




