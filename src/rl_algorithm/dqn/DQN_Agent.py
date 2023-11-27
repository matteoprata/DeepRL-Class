
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import torch.optim as optim

from src.rl_algorithm.dqn.DQN import DQN
from matplotlib import pyplot as plt


# DQN https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/tree/master
# https://docs.cleanrl.dev/rl-algorithms/dqn/#explanation-of-the-logged-metrics
# PPO https://github.com/xtma/pytorch_car_caring/tree/master


class DQNAgent:
    def __init__(self,
                 action_space,
                 epsilon=1.0,
                 gamma=0.95,
                 epsilon_min=0.1,
                 epsilon_decay=0.9999,
                 lr=1e-3,
                 memory_len=5000,
                 frames=3,
                 hidden_dimension=None,
                 device=None):

        self.device = device
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_len = memory_len
        self.lr = lr
        self.memory = deque(maxlen=self.memory_len)
        self.action_space = action_space

        self.target_model = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)
        self.model =        DQN(frames, len(self.action_space), hidden_dimension).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def is_explore(self):
        flip = np.random.rand() <= self.epsilon
        return flip

    def act(self, state=None, is_only_random=False):
        if self.is_explore() or is_only_random:
            action_index = np.random.randint(len(self.action_space))
            # print(action_index, self.action_space[action_index])
        else:
            q_values = self.target_model(state)[0]
            action_index = torch.argmax(q_values)
            # print("predicted action", action_index)
        return self.action_space[action_index]

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []

        for state, action_index, reward, next_state, done in minibatch:
            # state = torch.Tensor(state)
            target = self.model(state)[0]
            train_state.append(target)

            target_copy = target.detach().clone().to(self.device)   #  torch.Tensor(target).to(self.device)
            if done:
                target_copy[action_index] = reward
            else:
                t = self.target_model(next_state)[0]
                target_copy[action_index] = reward + self.gamma * torch.max(t)
            train_target.append(target_copy)

        # Actual training
        criterion = nn.MSELoss()
        pred, tru = torch.stack(train_state), torch.stack(train_target)
        loss = criterion(pred, tru)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, name):
        self.model = torch.load(name)
        self.target_model = torch.load(name)
        self.model.eval()

    def save_model(self, name):
        torch.save(self.target_model, name)
