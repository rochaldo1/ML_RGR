from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QGroupBox
)
from PyQt5.QtCore import Qt

from cartpole.cartpole_hill_climbing import train_hill_climbing
from cartpole.cartpole_policy_gradient import train_policy_gradient
from mountaincar.mountaincar_policy_gradient import train_mountaincar
from utils.plots import plot_rewards


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RL Training GUI (PyQt, OOP)")
        self.setGeometry(200, 200, 400, 350)

        layout = QVBoxLayout()

        # --- Hill Climbing Box ---
        self.hc_box = QGroupBox("Hill Climbing (CartPole)")
        hc_layout = QVBoxLayout()

        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise level:"))
        self.hc_noise = QLineEdit("0.1")
        noise_row.addWidget(self.hc_noise)
        hc_layout.addLayout(noise_row)

        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("Episodes:"))
        self.hc_episodes = QLineEdit("300")
        ep_row.addWidget(self.hc_episodes)
        hc_layout.addLayout(ep_row)

        btn = QPushButton("Run Hill Climbing")
        btn.clicked.connect(self.run_hill_climbing)
        hc_layout.addWidget(btn)

        self.hc_box.setLayout(hc_layout)
        layout.addWidget(self.hc_box)

        # --- Policy Gradient Box ---
        self.pg_box = QGroupBox("Policy Gradient (CartPole)")
        pg_layout = QVBoxLayout()

        ep_row2 = QHBoxLayout()
        ep_row2.addWidget(QLabel("Episodes:"))
        self.pg_episodes = QLineEdit("500")
        ep_row2.addWidget(self.pg_episodes)
        pg_layout.addLayout(ep_row2)

        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("Learning rate:"))
        self.pg_lr = QLineEdit("0.01")
        lr_row.addWidget(self.pg_lr)
        pg_layout.addLayout(lr_row)

        btn2 = QPushButton("Run Policy Gradient")
        btn2.clicked.connect(self.run_policy_gradient)
        pg_layout.addWidget(btn2)

        self.pg_box.setLayout(pg_layout)
        layout.addWidget(self.pg_box)

        # --- MountainCar PG Box ---
        self.mc_box = QGroupBox("MountainCar Policy Gradient")
        mc_layout = QVBoxLayout()

        btn3 = QPushButton("Run MountainCar PG")
        btn3.clicked.connect(self.run_mountaincar)
        mc_layout.addWidget(btn3)

        self.mc_box.setLayout(mc_layout)
        layout.addWidget(self.mc_box)

        # Set main layout
        self.setLayout(layout)

    # -------- Methods -----------

    def run_hill_climbing(self):
        noise = float(self.hc_noise.text())
        episodes = int(self.hc_episodes.text())

        _, rewards = train_hill_climbing(noise_level=noise, episodes=episodes)
        plot_rewards(rewards, f"Hill Climbing (noise={noise})")

    def run_policy_gradient(self):
        episodes = int(self.pg_episodes.text())
        lr = float(self.pg_lr.text())

        _, rewards = train_policy_gradient(episodes=episodes, lr=lr)
        plot_rewards(rewards, f"Policy Gradient (CartPole)")

    def run_mountaincar(self):
        _, rewards = train_mountaincar()
        plot_rewards(rewards, "MountainCar Policy Gradient")
