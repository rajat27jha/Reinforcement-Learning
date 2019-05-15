import gym
from agents import RandomAgent
from agents import ValueIterAgent
import numpy as np
env = gym.make('FrozenLake-v0') # in this we will solve Frozen lake problem, for more see documentation
gamma = 1

# Step1: Instantiate a Random/ValueIter Agent
# first we will instantiate random agent from agents.py
# agent = RandomAgent(env.action_space) # each environment(here Frozen lake) comes with its own action space

agent = ValueIterAgent(env, gamma) # this will not be random and will generate a value for each block of the game
agent.value_iteration()

# Step2: For Value Iter Agent, Evaluate Policy

agent.extract_policy()
print('Agent policy', agent.policy)

# Step3: Play Frozen Lake 1000 times with this policy and measure rewards

# playing 1000 times with with random stratergy
all_rewards = [] # keeping track of all rewards

for episode in range(1000):
    obs = env.reset() # taking previous observation
    total_rewards = 0
    while True:
        action = agent.choose_action(obs)  # callin method of random agent class in agents.py for taking a random action
        obs, reward, done, info = env.step(action)  # step method returns four values
        if done:
            all_rewards.append(reward)
            break
# Step4: Print Average Reward
print('Average Reward: ', np.mean(all_rewards)) # getting very less  i.e 0.019
# every time we fall we get a reward of 0 otherwise 1
