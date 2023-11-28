
import os
from datetime import datetime
import cv2
import torch
import matplotlib.pyplot as plt
import random
import numpy as np

import json


def write_json_to_file(data, file_path):
    """
    Write JSON data to a file.

    Parameters:
    - data: A dictionary representing the JSON data.
    - file_path: The path where the JSON file will be written.
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"JSON data successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing JSON data to {file_path}: {e}")


def read_json_from_file(file_path):
    """
    Read JSON data from a file.

    Parameters:
    - file_path: The path of the JSON file to be read.

    Returns:
    - A dictionary representing the JSON data.
    - If there is an error reading the file, returns None.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        print(f"JSON data successfully read from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading JSON data from {file_path}: {e}")
        return None


def make_all_paths(is_dynamic_root=True, dir_name="rl_class"):
    ROOT = "data"

    if is_dynamic_root:
        date_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        dir_name = "rl_class_{}".format(date_str)
    else:
        dir_name = dir_name

    path_root = ROOT + "/" + dir_name + "/"
    dirs = ["models", "plots", "videos"]
    for d in dirs:
        path = path_root + d
        if not os.path.exists(path):
            os.makedirs(path)
        print(">> Created dir", path)
    return path_root


def plot_state_car(data, title=None):
    assert len(data.shape) == 3, "Can only handle 3D mats."
    assert data.shape[0] < 10, "Too many states to plot. Adjust the plots position first."

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, data.shape[0], figsize=(10, 4))

    # Plot each image using imshow()
    for i in range(data.shape[0]):
        axs[i].imshow(data[i], cmap='gray')  # You can adjust the colormap if needed
        axs[i].axis('off')                   # Turn off axis labels

    plt.title(title)
    plt.show()


def plot_frame_car(data, title=None):
    plt.imshow(data, cmap="gray")  # You can adjust the colormap if needed
    plt.axis('off')  # Turn off axis labels
    plt.title(title)
    plt.show()


def preprocess_frame_car(frame):
    def crop(frame):
        # Crop to 84x84
        return frame[:-12, 6:-6]

    def make_img_gray(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def normalize(frame):
        return frame / 255.0

    # frame = crop(frame)
    frame = make_img_gray(frame)
    frame = frame.astype(float)
    frame = normalize(frame)
    # frame = frame * 2 - 1   # maps [0,1] to [-1,1]
    return frame


def seed_everything(seed=42):
    # Set seed for Python random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable if deterministic mode is desired
