
class Config:

    SEED = 1

    STARTING_EPISODE_TRAIN = 0
    ENDING_EPISODE_TRAIN = STARTING_EPISODE_TRAIN + 1000

    STARTING_EPISODE_TEST = ENDING_EPISODE_TRAIN + 1
    ENDING_EPISODE_TEST = STARTING_EPISODE_TEST + 100

    SKIP_FRAMES = 2
    TRAINING_BATCH_SIZE = 64
    UPDATE_TARGET_MODEL_FREQUENCY = 5
    N_FRAMES = 3
    HIDDEN_DIMENSION_FC = 150
    action_space = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)
    ]
    GAS_WEIGHT = 1.3
