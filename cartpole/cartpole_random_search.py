import gymnasium as gym
import numpy as np
from env.env_utils import evaluate_policy

def random_policy(theta, state):
    return 0 if np.dot(theta, state) < 0 else 1

def train_random_search(episodes=2000):
    env = gym.make("CartPole-v1")
    best_reward = 0
    best_theta = np.random.randn(4)

    for episode in range(episodes):
        theta = np.random.randn(4)
        reward = evaluate_policy(env, lambda s: random_policy(theta, s), episodes=1)

        if reward > best_reward:
            best_reward = reward
            best_theta = theta

    return best_theta, best_reward
