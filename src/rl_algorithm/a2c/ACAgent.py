import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import torch.optim as optim

from torch.distributions import Beta
from rl_algorithm.dqn.DQN import DQN
from matplotlib import pyplot as plt

from src.rl_algorithm.a2c.ACNet import ACNet


class ACAgent:
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self,
                 action_space,
                 epsilon=1.0,
                 gamma=0.95,
                 epsilon_min=0.1,
                 epsilon_decay=0.9999,
                 lr=1e-3,
                 memory_len=5000,
                 frames=3,
                 hidden_dimensions=None,
                 device=None):

        self.device = device
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_len = memory_len
        self.lr = lr
        self.action_space = action_space

        self.memory = deque(maxlen=self.memory_len)
        self.model = ACNet(frames, len(self.action_space), hidden_dimensions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.training_step = 0

    def act(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.model(state)[0]

        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def load_model(self, name):
        model = torch.load(name)
        model.eval()
        return model

    def save_model(self, name):
        torch.save(self.model, name)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def replay(self, batch_size):
        # self.training_step += 1
        #
        # s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        # a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        # r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        # s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        #
        # old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []

        for state, action_index, old_a_logp, reward, next_state, done in minibatch:

            with torch.no_grad():
                target_v = reward + self.gamma * self.model(next_state)[1]
                adv = target_v - self.model(state)[1]
                # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(self.ppo_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                    alpha, beta = self.model(state[index])[0]
                    dist = Beta(alpha, beta)
                    a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                    ratio = torch.exp(a_logp - old_a_logp[index])

        surr1 = ratio * adv[index]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
        action_loss = - torch.min(surr1, surr2).mean()
        value_loss = nn.SmoothL1Loss(self.model(state[index])[1], target_v[index])
        loss = action_loss + 2. * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


