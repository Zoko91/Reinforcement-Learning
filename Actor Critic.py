import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym

env = gym.make("CartPole-v1")
observation_dimension = env.observation_space.shape[0]
n_acts = env.action_space.n

# NeuralNet for the policy
net_stochastic_policy = nn.Sequential(
        nn.Linear(observation_dimension, 32),
        nn.Tanh(),
        nn.Linear(32, n_acts)
        )

# NeuralNet for the value function
net_value_function = nn.Sequential(
        nn.Linear(observation_dimension, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
        )

# Hyperparameters
learning_rate = 1e-2
epochs = 50
batch_size = 5000

# Optimizers
optimizer_policy = Adam(net_stochastic_policy.parameters(), lr=learning_rate)
optimizer_value = Adam(net_value_function.parameters(), lr=learning_rate)


def policy(observation):
    # Returns a distribution of "logits/probabilities" over actions
    logits = net_stochastic_policy(observation)
    return Categorical(logits=logits)


def choose_action(observation):
    # Returns an action sampled from the policy
    return policy(observation).sample().item()


def compute_loss(batch_observations, batch_actions, batch_weights):
    # Computes the loss for the policy
    batch_log_probability = policy(batch_observations).log_prob(batch_actions)
    return -(batch_log_probability * batch_weights).mean()


def compute_value_loss(batch_observations, batch_targets):
    # Computes the loss for the value function
    batch_values = net_value_function(batch_observations).squeeze(1)
    # Mean squared error loss
    return -((batch_values - batch_targets) ** 2).mean()


def compute_advantages(values, rewards, gamma=0.90):
    # Advantage is reward + gamma * V(next_state) - V(state)
    advantages = np.zeros(len(rewards))
    for t in range(len(rewards) - 1):
        advantages[t] = rewards[t] + gamma * values[t + 1] - values[t]
    return advantages


def actor_critic():
    # Train the policy and the value function
    for epoch in range(epochs):
        batch_observations = []
        batch_actions = []
        batch_returns = []
        batch_lengths = []

        observation = env.reset()[0]
        done = False
        rewards_in_episode = []

        while True:
            batch_observations.append(observation.copy())

            action = choose_action(torch.as_tensor(observation, dtype=torch.float32))
            observation, reward, done, _, _ = env.step(action)

            batch_actions.append(action)
            rewards_in_episode.append(reward)

            if done:
                episode_return = sum(rewards_in_episode)
                episode_length = len(rewards_in_episode)
                batch_returns.extend([episode_return] * episode_length)
                batch_lengths.append(episode_length)

                observation, done = env.reset()[0], False
                rewards_in_episode = []

                if len(batch_observations) > batch_size:
                    break

        # Calculate values predicted by the critic
        batch_values = net_value_function(torch.as_tensor(batch_observations, dtype=torch.float32)).squeeze(1)

        # Calculate advantages
        batch_advantages = torch.tensor(compute_advantages(batch_values.detach().numpy(),
                                                           np.array(batch_returns)), dtype=torch.float32)

        # Update the value function (critic)
        optimizer_value.zero_grad()
        value_loss = compute_value_loss(torch.as_tensor(batch_observations, dtype=torch.float32),
                                        torch.tensor(batch_returns, dtype=torch.float32))
        value_loss.backward()
        optimizer_value.step()

        # Update the policy (actor) with advantages
        optimizer_policy.zero_grad()
        batch_loss = compute_loss(torch.as_tensor(batch_observations, dtype=torch.float32),
                                  torch.as_tensor(batch_actions, dtype=torch.int64),
                                  batch_advantages)
        batch_loss.backward()
        optimizer_policy.step()

        print('epoch: %3d \t return: %.3f' % (epoch, np.mean(batch_returns)))


actor_critic()


def run_episode(env):
    obs = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        action = choose_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


policy_scores = [run_episode(env) for _ in range(100)]
print("Average score of the policy: ", np.mean(policy_scores))
