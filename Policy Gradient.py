import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import random

env = gym.make("CartPole-v1")
observation_dimension = env.observation_space.shape[0]
n_acts = env.action_space.n

#############################################
####### BUILDING A NEURAL NETWORK ###########
##### REPRESENTING A STOCHASTIC POLICY ######
#############################################

# net_stochastic_policy is a neural network representing a stochastic policy:
# it takes as inputs observations and outputs logits for each action
net_stochastic_policy = nn.Sequential(
        nn.Linear(observation_dimension, 32), 
        nn.Tanh(),
        nn.Linear(32, n_acts)
        )


# policy inputs an observation and computes a distribution on actions
def policy(observation):
    logits = net_stochastic_policy(observation)
    return Categorical(logits=logits)


# choose an action (outputs an int sampled from policy)
def choose_action(observation):
    return policy(observation).sample().item()


# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(batch_observations, batch_actions, batch_weights):
    batch_log_probability = policy(batch_observations).log_prob(batch_actions)
    return -(batch_log_probability * batch_weights).mean()


# Constants for training
learning_rate = 1e-2
epochs = 50
batch_size = 5000
########################

# make optimizer
optimizer = Adam(net_stochastic_policy.parameters(), lr=learning_rate)

#############################################
######### VANILLA POLICY GRADIENT ###########
#############################################


def vanilla_policy_gradient():
    for i in range(epochs):
        batch_observations = [] 
        batch_actions = []      
        batch_weights = []      
        batch_returns = []      
        batch_lengths = []      

        observation = env.reset()[0]
        done = False            
        rewards_in_episode = []            # list for rewards in the current episode

        # First step: collect experience by simulating the environment with current policy
        while True:
            batch_observations.append(observation.copy())

            # act in the environment
            action = choose_action(torch.as_tensor(observation, dtype=torch.float32))
            observation, reward, done, _, _ = env.step(action)

            # save action, reward
            batch_actions.append(action)
            rewards_in_episode.append(reward)

            if done:
                # if episode is over, record info about episode
                episode_return = sum(rewards_in_episode)
                episode_length = len(rewards_in_episode)
                batch_returns.append(episode_return)
                batch_lengths.append(episode_length)

                # the weight for each log probability (action | observation) is the total reward in the episode
                batch_weights += [episode_return] * episode_length

                # reset episode-specific variables
                observation, done = env.reset()[0], False
                rewards_in_episode = []

                # end experience loop if we have enough of it
                if len(batch_observations) > batch_size:
                    break

        # Step second: update the policy
        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(torch.as_tensor(batch_observations, dtype=torch.float32),
                                  torch.as_tensor(batch_actions, dtype=torch.int64),
                                  torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t episode_length: %.3f'%
                (i, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))


vanilla_policy_gradient()

###### EVALUATION ############


def run_episode(env, render=False):
    obs = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = choose_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


policy_scores = [run_episode(env) for _ in range(100)]
print("Average score of the policy: ", np.mean(policy_scores))

# for _ in range(2):
#   run_episode(env, True)

env.close()
