
import cv2   # open cv
import torch
from matplotlib import pyplot as plt
from src.rl_algorithm.dqn.DQN_Agent import DQNAgent
import gym
from collections import deque
import numpy as np
from gym.wrappers import RecordVideo
import random
import src.util as util
from src.config import Config

if __name__ == '__main__':

    util.seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('>> Using device:', device)

    agent = DQNAgent(frames=Config.N_FRAMES, action_space=Config.action_space, device=device, hidden_dimension=Config.HIDDEN_DIMENSION_FC)

    # https://www.gymlibrary.dev/environments/box2d/car_racing/
    env = gym.make('CarRacing-v2', render_mode="human")  # , render_mode='human')
    agent.load_model("data/working_models/trial_660.h5")

    for e in range(Config.STARTING_EPISODE, Config.ENDING_EPISODE + 1):
        env.episode_id = e

        init_state = env.reset()[0]  # 96, 96, 3 pixels image RGB
        init_state = util.preprocess_frame_car(init_state)  # 96, 96 pixels image GRAY

        # (1) EVALUATE STATE: S
        state_queue = deque([init_state] * Config.N_FRAMES, maxlen=Config.N_FRAMES)
        state_tensor = torch.Tensor(state_queue).unsqueeze(0).to(device)

        while True:
            state_tensor = torch.Tensor(state_queue).unsqueeze(0).to(device)
            action = agent.act(state_tensor, is_only_exploit=True)
            # (2) EXECUTE ACTION (for several steps)
            # (3) EVALUATE S' STATE, REWARD
            for _ in range(Config.SKIP_FRAMES):
                next_state, _, epi_done, _, _ = env.step(action)
                if epi_done:
                    break


            next_state = util.preprocess_frame_car(next_state)
            next_state_queue = deque([frame for frame in state_queue], maxlen=Config.N_FRAMES)
            next_state_queue.append(next_state)
            next_state_tensor = torch.Tensor(next_state_queue).unsqueeze(0).to(device)

            # S = S'
            state_queue = next_state_queue
        env.close()
