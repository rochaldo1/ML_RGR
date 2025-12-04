import gym
import numpy as np

def make_env(env_name):
    return gym.make(env_name, render_mode="human")

def evaluate_policy(env, policy, episodes=5):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total = 0
        done = False
        while not done:
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
    return np.mean(rewards)
