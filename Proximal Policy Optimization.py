"""
Joseph BEASSE / Enseirb-Matmeca IA 3A
Subject: PPO (proximal policy optimization) in RL

Environment: CartPole V1
--> https://www.gymlibrary.dev/environments/classic_control/cart_pole/

Resources: Proximal Policy Optimization Algorithms
     How to implement a PPO algorithm
 --> https://arxiv.org/abs/1707.06347
     Explanation on the clipped surrogate function and PPO
 --> https://huggingface.co/learn/deep-rl-course/unit8/introduction
     DeepRL foundation, TRPO and PPO implementation with basic algorithmic
 --> https://www.youtube.com/watch?v=KjWF8VIMGiY&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0&index=4&ab_channel=PieterAbbeel

Summary:
The following code depicts a provided Proximal Policy Optimization (PPO) implementation in python.
It is an RL algorithm for training a policy (a policy is a strategy or a set of rules that an agent uses to make
decisions in an environment) in the CartPole-v1 environment.

The stochastic policy is iteratively optimized by a neural network using PPO for 50 epochs.
The training loop collects observations, actions, and returns (rewards) over multiple episodes.
It then computes advantages (difference between the observed/actual return and the expected one)
and then applies the PPO surrogate loss with a clipped objective function.
The objective is to balance exploration and exploitation while preventing large policy changes
that could lead to instability. (the clipped part)

The final evaluation (100 episodes) indicates the average score of the policy, providing insight into its effectiveness
in balancing the CartPole environment.
"""


# Imports
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym

# Loading the environment
"""
CartPole-v1 is a classic reinforcement learning environment where the goal is to balance a pole on a moving cart. 
The agent receives a reward for each time step the pole remains upright, aiming to maximize cumulative rewards. 
Episodes end if the pole falls too much, the cart goes beyond a range, or a time limit is reached. 
"""
env = gym.make("CartPole-v1")
observation_dimension = env.observation_space.shape[0]
n_acts = env.action_space.n

# NeuralNet for the policy
"""
Simple feedforward architecture with one hidden layer. 
The Tanh activation function introduces non-linearity, and the output layer produces logits for each action in the environment.
The stochastic policy is defined by the probability distribution over actions, 
which is determined by applying the softmax function to the logits during the policy evaluation.
"""
net_stochastic_policy = nn.Sequential(
    nn.Linear(observation_dimension, 32),
    nn.Tanh(),
    nn.Linear(32, n_acts)
)

# Hyperparameters
learning_rate = 1e-2
epochs = 50
batch_size = 10000
clip_ratio = 0.2  # Clip ratio for PPO (often denoted as ϶, with the clip range being: [1 — ϶, 1 + ϶])

# Optimizer
optimizer_policy = Adam(net_stochastic_policy.parameters(), lr=learning_rate)

"""
The policy function sets up and returns a probability distribution over actions based on the current observation, 
and the choose_action function samples an action from this distribution, representing the chosen action for the agent 
to take in the environment.
"""
def policy(observation):
    # Returns a distribution of "logits/probabilities" over actions
    logits = net_stochastic_policy(observation)
    return Categorical(logits=logits)


def choose_action(observation):
    # Returns an action sampled from the policy
    return policy(observation).sample().item()



def ppo():
    # Training loop of the policy using PPO
    max_len = 0
    for epoch in range(epochs): # Iterates over the epochs
        batch_observations = []
        batch_actions = []
        batch_returns = []
        batch_lengths = []

        observation = env.reset()[0] # Observation the current state of the environment
        done = False
        rewards_in_episode = []

        while True:
            batch_observations.append(observation.copy())

            action = choose_action(torch.as_tensor(observation, dtype=torch.float32)) # Samples an action from the stochastic policy based on the current observation
            observation, reward, done, truncated, _ = env.step(action) # Executes the selected action in the environment
            done |= truncated  # Episodes length were going beyond the condition of 500 which cause the program to
            # have exploding results and never stop.

            batch_actions.append(action)
            rewards_in_episode.append(reward)

            if done: # if the episode is complete, calculates the total return for the episode then resets the environment for the next episode.
                episode_return = sum(rewards_in_episode)
                episode_length = len(rewards_in_episode)
                max_len = max(max_len, episode_length)
                batch_returns.extend([episode_return] * episode_length)
                batch_lengths.append(episode_length)

                observation, done = env.reset()[0], False
                rewards_in_episode = []

                if len(batch_observations) > batch_size:
                    break  # Exits the loop if out of batch

        # PPO Training
        optimizer_policy.zero_grad()

        # Convert lists to tensors
        batch_observations = torch.as_tensor(np.vstack(batch_observations), dtype=torch.float32)
        batch_actions = torch.as_tensor(batch_actions, dtype=torch.int64)
        batch_returns = torch.tensor(batch_returns, dtype=torch.float32)

        # Compute advantages
        with torch.no_grad():
            old_policy = policy(batch_observations)
            old_log_prob = old_policy.log_prob(batch_actions)
            old_values = net_stochastic_policy(batch_observations).max(1)[0]
            # Advantages represent the difference between the actual returns obtained and the values predicted by the policy
            # Should not be counted as operations for the gradient descend, thus the torch.no_grad()
            advantages = batch_returns - old_values

        # The PPO Surrogate Loss is then computed
        for _ in range(10):  # PPO epoch (used to perform multiple optimization steps to improve stability)
            new_policy = policy(batch_observations)
            new_log_prob = new_policy.log_prob(batch_actions)

            ratio = (new_log_prob - old_log_prob).exp() # Calculates the ratio of probabilities between the new policy and the old policy
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) # Clamping the ratio within a specified range. This step helps prevent overly large policy updates.

            # PPO objective function
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages) # Surrogate loss used to update the policy
            optimizer_policy.zero_grad()
            loss.mean().backward() # backpropagation to calculate the gradients of the neural network parameters
            optimizer_policy.step() # update of the parameters based on the computed gradients

        # For the current training epoch:
        print('epoch: %3d \t return: %.3f' % (epoch+1, torch.mean(batch_returns).item()))

# Initializes the stochastic policy network and trains the model using the PPO implementation
ppo()


def run_episode(env):
    """
    This function runs a single episode in the given environment using the learned policy.
    It is used to evaluate the performance of the trained policy after training.
    It chooses actions based on the policy,
    interacts with the environment,
    accumulates rewards,
    then returns the total reward obtained during the episode.
    """
    obs = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        """
        The episode is done if any of the following cases is obtained:
        Pole Angle is greater than ±12°
        Cart Position is greater than ±2.4 or
        Episode length is greater than 500
        While it is not done, chooses an action and accumulates the reward.
        """
        action = choose_action(torch.as_tensor(np.array(obs, dtype=np.float32)))
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        done |= truncated
        if done:
            break
    return total_reward


policy_scores = [run_episode(env) for _ in range(100)] # Obtains the score of 100 episodes
print("Average score of the policy: ", np.mean(policy_scores)) # Prints the average score of the policy over the 100 episodes

"""
To conclude, regarding the following results (below) the training process appears successful. 
The policy is achieving a high average score (near the maximum one) and showing consistent improvement over epochs. 
The obtained scores suggest that the policy has learned a robust and effective strategy for the CartPole task.
"""

# Maximum score: 500
# Average score of the policy:  445.97

# Values obtained by a 1 simulation:
# epoch:   1 	 return: 24.801
# epoch:  10 	 return: 180.949
# epoch:  20 	 return: 351.183
# epoch:  30 	 return: 376.672
# epoch:  40 	 return: 458.430
# epoch:  50 	 return: 475.142

