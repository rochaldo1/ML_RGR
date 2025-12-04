import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

def train_policy_gradient(episodes=500, lr=0.01):
    env = gym.make("CartPole-v1")
    policy = PolicyNN()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_rewards = []

    for ep in range(episodes):
        states, actions, rewards = [], [], []

        state, _ = env.reset()
        done = False

        while not done:
            s = torch.tensor(state, dtype=torch.float32)
            probs = policy(s)
            action = torch.multinomial(probs, 1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(s)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # REINFORCE Update
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        loss = 0
        for s, a, G in zip(states, actions, returns):
            probs = policy(s)
            log_prob = torch.log(probs[a])
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(sum(rewards))

    return policy, all_rewards
