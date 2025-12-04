import gym
import numpy as np
from env.env_utils import evaluate_policy

def policy(theta, state):
    return 0 if np.dot(theta, state) < 0 else 1

def train_hill_climbing(noise_level=0.1, episodes=300):
    env = gym.make("CartPole-v1")
    theta = np.random.randn(4)
    best_reward = evaluate_policy(env, lambda s: policy(theta, s))

    rewards = []

    for ep in range(episodes):
        theta_new = theta + noise_level * np.random.randn(4)
        reward = evaluate_policy(env, lambda s: policy(theta_new, s))
        rewards.append(reward)

        if reward > best_reward:
            best_reward = reward
            theta = theta_new

    return theta, rewards
