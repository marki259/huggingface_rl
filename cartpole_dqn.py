# Refer to https://github.com/lmarza/CartPole-CNN/blob/main/dqnCartPole.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

import numpy as np
import random

from collections import deque
from typing import Tuple


class DQN(nn.Module):
    def __init__(self, space_size: int = 4, n_hidden: int = 128, action_size: int = 2):
        super(DQN, self).__init__()

        self.dense1 = nn.Linear(space_size, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, action_size)

        self.relu = nn.ReLU()

        self.loss = nn.MSELoss()

        self.memory = deque(maxlen=10000)

        self.space_size = space_size
        self.action_size = action_size

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Hyperparameters
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.batch_size = 32

        self.gamma = 0.95

        self.scores = []
        self.episodes = []
        self.average = []

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.output(x)
        
        return x

    def remember(self, state_tuple: Tuple):
        self.memory.append(state_tuple)

    def act(self, state):
        self.eval()

        if np.random.uniform() <= self.epsilon:
            action = np.random.choice(range(self.action_size))
        else:
            action = torch.argmax(self.forward(state)).item()

        return action

    def replay(self):
        self.train()

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in mini_batch])
        rewards = np.array([t[2] for t in mini_batch])
        actions = np.array([t[1] for t in mini_batch])
        next_states = np.array([t[3] for t in mini_batch])
        truncations = np.array([t[4] for t in mini_batch])
        terminations = np.array([t[5] for t in mini_batch])

        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        terminations = torch.Tensor(terminations).unsqueeze(1)
        rewards = torch.Tensor(rewards).unsqueeze(1)
        actions = torch.Tensor(actions).squeeze(0).long()

        idx = torch.Tensor([i for i in range(len(mini_batch))]).long()

        self.optimizer.zero_grad()

        G_current = self(states).squeeze(1)
        G_next, _ = self(next_states).max(dim=2)
        G_next *= (1.0 - terminations)
        targets = rewards + self.gamma * G_next
        G_approx = G_current.clone()
        G_current[idx, actions] = targets.squeeze(1)
        
        loss = self.loss(G_approx, G_current)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def fit(self, n_episodes: int) -> list:

        for i in range(n_episodes):
            state, _ = env.reset()
            state = torch.Tensor(state).unsqueeze(0)
            score = 0
            max_steps = 1000

            for j in range(max_steps):
                action = self.act(state)

                next_state, reward, truncated, terminated, _ = env.step(action)
                next_state = torch.Tensor(next_state).unsqueeze(0)

                self.remember((state.detach().numpy(), action, reward, next_state, truncated, terminated))
                score += reward
                state = next_state
                if len(self.memory) >= self.batch_size:
                    self.replay()
                if truncated or terminated:
                    self.scores.append(score)
                    break

            if (i % 10) == 0:
                print("Mean score at {}: {:.2f}".format(i, np.mean(self.scores[-10:])))

            if np.mean(self.scores[-10:]) > 120:
                break
                
        return self.scores
                    


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    ep = 300

    dqn_model = DQN()
    dqn_model.fit(ep)
