import numpy as np
import random
from environment import Environment
from agents import RandomAgent
from agents import ValueApproxAgent

env = Environment(5)
agent = RandomAgent(env.action_space)

total_rewards = 0
for i in range(1000):
    action = agent.choose_action()
    reward = env.try_arm(action)
    total_rewards += reward

print('Total Rewards From RandomAgent: ', total_rewards)

agent = ValueApproxAgent(env.action_space)
total_rewards = 0
for i in range(1000):
    action = agent.choose_action()
    reward = env.try_arm(action)
    print('action: ', action)
    agent.learn(action, reward)
    total_rewards += reward

print('Total Rewards From ValueApproxAgent: ', total_rewards)
print(agent.approx_values, env._probs)