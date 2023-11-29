
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
import argparse


def test_car_racing(model_to_load):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('>> Using device:', device)

    # https://www.gymlibrary.dev/environments/box2d/car_racing/
    env = gym.make('CarRacing-v2', render_mode="human")  # , render_mode='human')
    util.seed_everything(seed=Config.SEED)

    agent = DQNAgent(frames=Config.N_FRAMES, action_space=Config.ACTION_SPACE, device=device,
                     hidden_dimension=Config.HIDDEN_DIMENSION_FC)

    agent.load_model(model_to_load)

    PICKED_EPISODES = [1]
    for e in PICKED_EPISODES:
        env.episode_id = e

        init_state = env.reset(seed=e)[0]                   # 96, 96, 3 pixels image RGB
        init_state = util.preprocess_frame_car(init_state)  # 96, 96 pixels image GRAY

        # (1) EVALUATE STATE: S
        # [S0, S0, S0] > [S1, S0, S0] > [S2, S1, S0] > [S3, S2, S1]
        state_queue = deque([init_state] * Config.N_FRAMES, maxlen=Config.N_FRAMES)
        epi_n_neg_rew = 0  # counts consecutive times agent got negative rewards

        while True:
            state_tensor = torch.Tensor(np.array(state_queue)).unsqueeze(0).to(device)
            action = agent.act(state_tensor, is_only_exploit=True)  # queries \pi without exploring
            # (2) EXECUTE ACTION (for several steps)
            # (3) EVALUATE S' STATE, REWARD

            reward = 0
            for _ in range(Config.SKIP_FRAMES):
                next_state, r, epi_done, _, _ = env.step(action)
                reward += r
                if epi_done:
                    break

            epi_n_neg_rew = epi_n_neg_rew + 1 if reward <= 0 else 0  # counts consecutive negative rewards
            if epi_n_neg_rew >= 100:  # took a wrong path, interrupt
                break

            next_state = util.preprocess_frame_car(next_state)
            next_state_queue = deque([frame for frame in state_queue], maxlen=Config.N_FRAMES)
            next_state_queue.append(next_state)

            # S = S'
            state_queue = next_state_queue
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mod', help='Pretrained model file.', required=False, default="data/working_models/trial_660.h5")
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    test_car_racing(model_to_load=args['mod'])
