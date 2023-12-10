import cozmo
import cv2
import numpy as np
import torch.optim as optim
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


import cozmo
import torchvision.transforms as transforms

import time
import os
from pynput import keyboard
import csv
from statistics import median
from collections import deque
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
def save_replay_buffer(replay_buffer, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(replay_buffer, f)

def load_replay_buffer(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def process_image(image):
    # Remove the extra dimension for single channel images
    if image.shape[0] == 1:
        image = image.squeeze(0)
    
    # Convert to NumPy array
    if isinstance(image, torch.Tensor):
        image = image.numpy()
        # Scale and convert type if necessary
        image = np.array(image * 255, dtype=np.uint8)
    
    return image
    
replay_buffer_path = 'replay_buffer'

replay_buffer = ReplayBuffer(100000)
if os.path.exists(replay_buffer_path):
    replay_buffer = load_replay_buffer(replay_buffer_path)
    print(f"Length of RB: {len(replay_buffer)}")
    #print(replay_buffer.sample(1)[0][3].shape)
    sample = replay_buffer.sample(1)
    #print(f"Actions taken: {sample[0][1]}, Reward: {sample[0][2]}, Done: {sample[0][4]}")
    # Sample two images
    sampled_image1 = sample[0][0]  # Existing image
    sampled_image2 = sample[0][3]  # New image to display
    new_buffer = ReplayBuffer(100000)


    #print(replay_buffer.buffer[0][2])
    for i in range(0, len(replay_buffer.buffer)):
        input_tensor = replay_buffer.buffer[i][0]
        left_wheel, right_wheel = replay_buffer.buffer[i][1]
        if replay_buffer.buffer[i][2] == -100:
            reward = 0
        else:
            reward = replay_buffer.buffer[i][2] / 100
        n_state = replay_buffer.buffer[i][3]
        done = replay_buffer.buffer[i][4]
        # Convert to integer tensors
        # left_wheel_int = left_wheel.long()
        # right_wheel_int = right_wheel.long()

        #print(left_wheel, right_wheel)

        if left_wheel.item() > 2 or right_wheel.item() > 2:
            if len(input_tensor.shape) == 2:
                new_buffer.store((input_tensor.unsqueeze(0), (left_wheel,right_wheel), reward, n_state, done))
            else:
                new_buffer.store((input_tensor, (left_wheel,right_wheel), reward, n_state, done))


    # print(replay_buffer.buffer[0])
    print(len(new_buffer))

    save_replay_buffer(new_buffer,"merged_replay_buffer_2")

    # Process both images
    # processed_image1 = process_image(sampled_image1)
    # processed_image2 = process_image(sampled_image2)

    # # Display the first image with the title "State"
    # cv2.imshow("State", processed_image1)

    # # Display the second image with the title "New State"
    # cv2.imshow("New State", processed_image2)

    # cv2.waitKey(0)  # Wait for a key press to close the windows
    # cv2.destroyAllWindows()