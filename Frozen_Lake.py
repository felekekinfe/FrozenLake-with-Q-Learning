import gym
import numpy as np
import random

env=gym.make("FrozenLake-v1", is_slippery=True)

state_size=env.observation_space.n
action_size=env.action_space.n

q_table=np.zeros((state_size,action_size))


num_episodes=20
max_steps=10
learning_rate=0.8
discount_factor=0.95
epsilon=1.0
min_epsilon=0.01
epsilon_decay=0.995

reward_list=list()

for episode in range(num_episodes):
    state=env.reset()[0]
    total_reward=0
    done=False
 

    for step in range(max_steps):
        if random.uniform(0,1)<epsilon:
            action=env.action_space.sample()
            
        else:
            action=np.argmax(q_table[state, :])
        new_state,reward,done,_,_=env.step(action)
        q_table[state,action]=q_table[state,action]+learning_rate*(reward+discount_factor*np.max(q_table[new_state,:])-q_table[state,action])

        state=new_state
        total_reward+=reward

        if done:
            break
    epsilon=max(min_epsilon,epsilon*epsilon_decay)
    reward_list.append(total_reward)

    if (episode+1)%100==0:
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_rewards}, Epsilon: {epsilon:.3f}")

