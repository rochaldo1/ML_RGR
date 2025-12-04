import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

class MCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

def train_mountaincar(episodes=2000, lr=0.01):
    env = gym.make("MountainCar-v0")
    policy = MCPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    rewards_list = []

    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        states, actions, rewards = [], [], []

        while not done:
            s_tensor = torch.tensor(s, dtype=torch.float32)
            probs = policy(s_tensor)
            action = torch.multinomial(probs, 1).item()

            next_s, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(s_tensor)
            actions.append(action)
            rewards.append(r)

            s = next_s

        # REINFORCE
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        loss = 0
        for s, a, G in zip(states, actions, returns):
            loss -= torch.log(policy(s)[a]) * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_list.append(sum(rewards))

    return policy, rewards_list
