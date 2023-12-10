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


def find_line_center(contours):
    # Find the largest contour and its center
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
    else:
        cX = None
    return cX

def take_image_and_find_deviation(robot, threshold_value = 30, iterations=5, max_time=2):
    deviations = []
    start_time = time.time()
    while len(deviations)<iterations:
        # Capture new image after movement
        robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=5)
        image = robot.world.latest_image.raw_image.convert("RGB")

        # Convert PIL Image to numpy array

        image_np = np.array(image)
        image_np = image_np[100:image_np.shape[0]-50, 20:280]
        image_width = image_np.shape[1]

        bw = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, bw_thresh = cv2.threshold(bw, threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bw_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_center = image_width // 2
        if contours: 
            line_center = find_line_center(contours)

            # Calculate deviation from the center of the image if a line center was found
            if line_center is not None:
                dev = line_center - image_center
                if dev < 127:
                    deviations.append(dev)
        
        if time.time() - start_time > max_time:
            print("Warning: maximum wait time to get deviation exceeded\nwaiting 2 sec")
            #time.sleep(2)
            return 1000 # high deviation for going out of track

    return median(deviations)

def get_image(robot, threshold_value = 30):
    # Get the latest image from Cozmo's camera
    robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=5)
    pil_initial_image = robot.world.latest_image.raw_image.convert("RGB")
    # Convert PIL Image to numpy array
    initial_image_np = np.array(pil_initial_image)
    initial_image_np = initial_image_np[50:initial_image_np.shape[0]-50, 20:280]
    bw_thresh = cv2.cvtColor(initial_image_np, cv2.COLOR_BGR2GRAY)
    # Apply the threshold to make white whiter and black blacker
    _, bw = cv2.threshold(bw_thresh, threshold_value, 255, cv2.THRESH_BINARY_INV)
    input_tensor = transforms.functional.to_tensor(bw).unsqueeze(0)
    # print(input_tensor.shape,input_tensor.unsqueeze(0).shape)
    return input_tensor

def calculate_reward(deviation):
        if deviation == 1000:
            return 0
        if abs(deviation) < 20:
            deviation = 0
            
        reward = (1 / (1 + abs(deviation))) # inverse of deviation
        return reward


def get_wheel_speeds():
    """
    Prompt the user to enter the left and right wheel speeds.
    """
    try:
        left_wheel_speed = float(input("Enter left wheel speed: ")) 
        right_wheel_speed = float(input("Enter right wheel speed: ")) 
        return left_wheel_speed, right_wheel_speed
    except ValueError:
        print("Please enter valid numbers.")
        return None, None

def cozmo_drive(robot: cozmo.robot.Robot):
    """
    Drive Cozmo based on keyboard input.
    """
    replay_buffer_path = 'cozmo_replay'
    step = 0

    replay_buffer = ReplayBuffer(100000)
    if os.path.exists(replay_buffer_path):
        replay_buffer = load_replay_buffer(replay_buffer_path)
        print(f"Buffer Loaded!!!\nLength of RB: {len(replay_buffer)}")
        # print(f"{replay_buffer.buffer[-1]}")
        remove_last = 0
        for _ in range(remove_last):
                print(replay_buffer.buffer.pop())
                print(f"Buffer Edited!!!\nLength of RB: {len(replay_buffer)}")
        step = len(replay_buffer.buffer)

    robot.set_head_angle(cozmo.util.degrees(-25)).wait_for_completed()
    # Set Cozmo's camera to stream in color
    robot.camera.color_image_enabled = True
    start_flag = time.time()
    total_reward = 0.0
    while True:
        input_tensor = get_image(robot)
        #if abs(take_image_and_find_deviation(robot))<125:
        left_wheel_speed, right_wheel_speed = get_wheel_speeds()

        if left_wheel_speed is not None and right_wheel_speed is not None:
            print(f"Moving Cozmo - Left: {left_wheel_speed}, Right: {right_wheel_speed}")
            robot.drive_wheels(left_wheel_speed, right_wheel_speed, duration=1)
            time.sleep(1)
            deviation = take_image_and_find_deviation(robot)
            reward = calculate_reward(deviation)
            total_reward += reward
            step += 1
            done = deviation == 1000
            print(f"Reward: {reward}")
            if input(f"Set New Reward?\nPress 'y' to set new reward: ") == 'y':
                new_reward = float(input(f"Set Custom Reward: "))
                if 0.0<= new_reward <= 1.0:
                    print(f"Setting Custom Reward: {new_reward}....")
                    reward = new_reward
        

        # Add an option to break out of the loop
        if input(f"Terminal Step:{done}\nPress 'q' to quit, any other key to continue: ") == 'q':
            if input(f"Set Done?\nPress 'y' to set done:") == 'y':
                new_done = int(input(f"Done? (0 or 1): "))
                if new_done == 1:
                    done = True
                elif new_done == 0:
                    done = False
            print(f"Done: {done}")
            replay_buffer.store((input_tensor.squeeze(0), (torch.tensor([left_wheel_speed//10]), torch.tensor([right_wheel_speed//10])), reward, get_image(robot).squeeze(0), done))
            save_replay_buffer(replay_buffer,replay_buffer_path)
            break
        else:
            replay_buffer.store((input_tensor.squeeze(0), (torch.tensor([left_wheel_speed//10]), torch.tensor([right_wheel_speed//10])), reward, get_image(robot).squeeze(0), done))
            save_replay_buffer(replay_buffer,replay_buffer_path)

# Run the Cozmo program
cozmo.run_program(cozmo_drive, use_viewer=True)