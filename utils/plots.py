import matplotlib.pyplot as plt

def plot_rewards(rewards, title):
    plt.figure()
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()
    