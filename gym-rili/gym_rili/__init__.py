from gym.envs.registration import register


register(
    id='rili-circle-v0',
    entry_point='gym_rili.envs:Circle',
    max_episode_steps=10,
)

register(
    id='rili-circle-N-v0',
    entry_point='gym_rili.envs:Circle_N',
    max_episode_steps=10,
)

register(
    id='rili-driving-v0',
    entry_point='gym_rili.envs:Driving',
    max_episode_steps=10,
)

register(
    id='rili-robot-v0',
    entry_point='gym_rili.envs:Robot',
    max_episode_steps=10,
)
