import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

class MCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

def train_mountaincar(episodes=5000, lr=0.02):
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

            r += abs(next_s[1]) * 0.1 # награда за скорость
            r += (next_s[0] + 0.5) # награда за позицию

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

        # нормализация returns
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for s_t, a_t, G_t in zip(states, actions, returns):
            loss -= torch.log(policy(s_t)[a_t]) * G_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_list.append(sum(rewards))

        # вывод прогресса каждые 100 эпизодов
        if (ep + 1) % 100 == 0:
            avg_reward = sum(rewards_list[-100:]) / 100
            print(f"Episode {ep+1}/{episodes}, Average Reward: {avg_reward:.2f}")
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                app.processEvents()
            except:
                pass

    env.close()
    return policy, rewards_list
