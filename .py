import gym
import numpy as np
import random

env=gym.make("FrozenLake-v1", is_slippery=True)
state_size=env.observation_space.n
action_size=env.action_space.n

q_table=np.zeros((state_size,action_size))

print(q_table)

num_episodes=200
max_steps=10
learning_rate=0.8
discount_factor=0.95
epsilon=1.0
min_epsilon=0.01
epsilon_decay=0.995
