
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_frames, n_actions, h_dimension):
        super(DQN, self).__init__()

        # CNN
        self.layers_cnn = nn.Sequential(
            nn.Conv2d(n_frames, 6, kernel_size=(7, 7), stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(6, 12, kernel_size=(4, 4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(432, h_dimension),
            nn.ReLU(),
            nn.Linear(h_dimension, n_actions)
        )

    def forward(self, x):
        o = self.layers_cnn(x)  # (BS, ACTIONS)
        return o

