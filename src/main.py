
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


if __name__ == '__main__':

    util.seed_everything()
    PATH_ROOT = util.make_all_paths(is_dynamic_root=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('>> Using device:', device)

    RENDER = True
    STARTING_EPISODE = 1
    ENDING_EPISODE = 1000
    SKIP_FRAMES = 5
    TRAINING_BATCH_SIZE = 64
    SAVE_TRAINING_FREQUENCY = 10
    UPDATE_TARGET_MODEL_FREQUENCY = 10
    N_FRAMES = 4

    action_space = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0),  #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]

    agent = DQNAgent(frames=N_FRAMES, action_space=action_space, device=device)

    # https://www.gymlibrary.dev/environments/box2d/car_racing/
    env = gym.make('CarRacing-v2', render_mode="rgb_array")  # , render_mode='human')
    env = RecordVideo(env, PATH_ROOT + 'video', episode_trigger=lambda x: x % UPDATE_TARGET_MODEL_FREQUENCY == 0)

    epi_total_rewards = []
    for e in range(STARTING_EPISODE, ENDING_EPISODE + 1):
        env.episode_id = e

        init_state = env.reset()[0]                         # 96, 96, 3 pixels image RGB
        init_state = util.preprocess_frame_car(init_state)  # 96, 96 pixels image GRAY

        epi_total_reward = 0
        epi_negative_reward_counter = 0
        epi_time_frame_counter = 1
        epi_done = False

        # S0
        state_queue = deque([init_state] * N_FRAMES, maxlen=N_FRAMES)
        state_tensor = torch.Tensor(state_queue).unsqueeze(0).to(device)
        # util.plot_state_car(np.array(state_queue))  # visualize S0

        while True:
            action = agent.act(state_tensor)

            # execute action for several steps
            reward = 0
            for _ in range(SKIP_FRAMES):
                # execute action
                next_state, r, epi_done, info, _ = env.step(action)
                reward += r
                if epi_done:
                    break

            # if getting negative reward 10 times after the tolerance steps, terminate this episode
            if epi_time_frame_counter > 100 and reward < 0:
                epi_negative_reward_counter += 1
            else:
                epi_negative_reward_counter = 0

            # extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            epi_total_reward += reward

            # S'
            next_state = util.preprocess_frame_car(next_state)
            next_state_queue = deque(state_queue, maxlen=N_FRAMES)  # creates a new queue
            next_state_queue.append(next_state)
            next_state_tensor = torch.Tensor(next_state_queue).unsqueeze(0).to(device)

            # Memorizing saving state, action reward tuples
            agent.memorize(state_tensor, action, reward, next_state_tensor, epi_done)

            # S = S'
            state_queue = next_state_queue

            # early stop if the number of
            if epi_negative_reward_counter >= 25 or epi_total_reward < 0:
                break

            # train the model with tuple, if there are enough tuples
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)

            epi_time_frame_counter += 1
        epi_total_rewards += [epi_total_reward]

        # >>> ON EPISODE END
        # print stats
        stats_string = 'Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}, Epsilon: {:.2}'
        print(stats_string.format(
            e,
            ENDING_EPISODE,
            epi_time_frame_counter,
            float(epi_total_reward),
            float(agent.epsilon))
        )

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            # plot rewards stats
            plt.plot(epi_total_rewards, label="cum rew", color="blue")
            plt.title("Rewards during episode episode")
            plt.savefig(PATH_ROOT + 'plots/reward_{}.pdf'.format(e))

            # save model frequently
            agent.save_model(PATH_ROOT + 'models/trial_{}.h5'.format(e))

            # swap model
            agent.update_target_model()

        env.close()
