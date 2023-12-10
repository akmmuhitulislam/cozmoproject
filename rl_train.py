import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import torch.optim as optim
from PIL import Image

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

class CozmoDriveNet(nn.Module):
    def __init__(self):
        super(CozmoDriveNet, self).__init__()

        # CNN for image processing
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Example output size from CNN
        cnn_output_size = 63488  # Set this based on your CNN structure

        # Shared layers
        self.shared_fc1 = nn.Linear(in_features=cnn_output_size, out_features=1024)
        self.shared_fc2 = nn.Linear(1024, 256)

        # Left wheel head
        self.left_head_fc1 = nn.Linear(256, 32)
        self.left_head_fc2 = nn.Linear(32, 1)

        # Right wheel head
        self.right_head_fc1 = nn.Linear(256, 32)
        self.right_head_fc2 = nn.Linear(32, 1)

    # Number of discrete classes (0, 10, 20, ..., 200)
        num_classes = 21

        # Output heads
        self.left_head = nn.Linear(256, num_classes)
        self.right_head = nn.Linear(256, num_classes)

    def forward(self, x):
        # Process input through CNN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)

        # Shared layers
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        # Output heads with softmax
        left = F.softmax(self.left_head(x), dim=1)
        right = F.softmax(self.right_head(x), dim=1)

        return left, right

def update_target_model(online_model, target_model):
    target_model.load_state_dict(online_model.state_dict())

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

def take_image_and_find_deviation(robot, threshold_value = 30, iterations=3, max_time=2):
    deviations = []
    start_time = time.time()
    while len(deviations)<iterations:
        # Capture new image after movement
        robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=5)
        image = robot.world.latest_image.raw_image.convert("RGB")

        image_np = np.array(image)
        image_np = image_np[100:image_np.shape[0]-60, 20:280]
        image_width = image_np.shape[1]

        bw = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        _, bw_thresh = cv2.threshold(bw, threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bw_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_center = image_width // 2
        if contours: 
            line_center = find_line_center(contours)

            # Calculate deviation from the center of the image if a line center was found
            if line_center is not None:
                dev = line_center - image_center
                if dev < 125:
                    deviations.append(dev)
        
        if time.time() - start_time > max_time:
            #print("Warning: maximum wait time to get deviation exceeded\nwaiting 2 sec")
            #time.sleep(2)
            return 1000 # high deviation for going out of track

    return median(deviations)

def get_image(robot, threshold_value = 30):
    # Get image from Cozmo's camera
    robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=5)
    pil_initial_image = robot.world.latest_image.raw_image.convert("RGB")
    initial_image_np = np.array(pil_initial_image)
    initial_image_np = initial_image_np[50:initial_image_np.shape[0]-50, 20:280]
    bw_thresh = cv2.cvtColor(initial_image_np, cv2.COLOR_BGR2GRAY)
    # apply threshold 
    _, bw = cv2.threshold(bw_thresh, threshold_value, 255, cv2.THRESH_BINARY_INV)
    input_tensor = transforms.functional.to_tensor(bw).unsqueeze(0)
    # print(input_tensor.shape,input_tensor.unsqueeze(0).shape)
    return input_tensor

def calculate_reward(deviation):
        if deviation == 1000:
            return 0
        reward = (1 / (1 + abs(deviation))) 
        return reward

def update_dqn(model, target_model, batch, optimizer, device, gamma=0.95):
    if len(batch) == 0:
        return 0  

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states).to(device)
    next_states = torch.stack(next_states).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)  
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)

    # Current Q values 
    current_q_probs_left, current_q_probs_right = model(states)
    current_q_values_left = current_q_probs_left.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1)
    current_q_values_right = current_q_probs_right.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1)

    # Next Q values 
    next_q_probs_left, next_q_probs_right = target_model(next_states)
    max_next_q_values_left = next_q_probs_left.detach().max(1)[0]
    max_next_q_values_right = next_q_probs_right.detach().max(1)[0]

    # # Zero-out terminal states
    # max_next_q_values_left[dones] = 0  
    # max_next_q_values_right[dones] = 0  

    # target Q values
    target_q_values_left = rewards + (gamma * max_next_q_values_left)
    target_q_values_right = rewards + (gamma * max_next_q_values_right)

    # loss
    loss_left = F.mse_loss(current_q_values_left, target_q_values_left)
    loss_right = F.mse_loss(current_q_values_right, target_q_values_right)
    loss = loss_left + loss_right

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def select_action(model, state, epsilon, num_classes=21):
    if random.random() > epsilon:
        # Exploitation
        with torch.no_grad():
            left_probs, right_probs = model(state)
        print(torch.argmax(left_probs, dim=1),  torch.argmax(right_probs, dim=1))
    else:
        # Exploration
        with torch.no_grad():
            left_probs, right_probs = model(state)
        
        left_action = torch.argmax(left_probs, dim=1).item()
        right_action = torch.argmax(right_probs, dim=1).item()

        # Adding noise to actions
        noise_scale = 5
        left_action_noise = random.randint(-(noise_scale-2), noise_scale)
        right_action_noise = random.randint(-(noise_scale-2), noise_scale)

        left_action = max(0, min(num_classes - 1, left_action + left_action_noise))
        right_action = max(0, min(num_classes - 1, right_action + right_action_noise))

        left_probs = torch.nn.functional.one_hot(torch.tensor(left_action), num_classes).float().unsqueeze(0)
        right_probs = torch.nn.functional.one_hot(torch.tensor(right_action), num_classes).float().unsqueeze(0)
        print("noisy actions")
        print(torch.argmax(left_probs, dim=1),  torch.argmax(right_probs, dim=1))


    return left_probs, right_probs

def cozmo_program(robot: cozmo.robot.Robot):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path, replay_buffer_path = 'cozmo_model_10' , 'replay_buffer'
    #model_path, replay_buffer_path = 'cozmo_model.pth', 'cozmo_replay'
    #target_model_path = 'cozmo_target_model.pth'
    csv_path = 'training_data.csv'
    
    model = CozmoDriveNet().to(device)
    target_model = CozmoDriveNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(100000)

    step = 0

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        target_model.load_state_dict(model.state_dict())
    # if os.path.exists(target_model_path):
    #     target_model.load_state_dict(torch.load(target_model_path))

    if os.path.exists(replay_buffer_path):
        replay_buffer = load_replay_buffer(replay_buffer_path)
        # replay_buffer2 = load_replay_buffer('cozmo_replay_t')
        # replay_buffer.buffer.append(x for x in replay_buffer.buffer)
        # replay_buffer.buffer.append(x for x in replay_buffer2.buffer)
        remove_last = 0
        for _ in range(remove_last):
                print(replay_buffer.buffer.pop())
        step = len(replay_buffer.buffer)

    # Exploration/Exploitation tradeoff
    epsilon_start = 0.00 # set this value to perform exploration
    epsilon_end = 0.0
    epsilon_decay = 0.90
    epsilon = epsilon_start


    # Set up the listener for the 'q' key
    exit_flag = False
    def on_press(key):
        nonlocal exit_flag
        if key == keyboard.Key.esc or (hasattr(key, 'char') and key.char == 'q'):
            exit_flag = True
            return False  

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Robot settings
    robot.set_head_angle(cozmo.util.degrees(-25)).wait_for_completed()
    robot.set_lift_height(0).wait_for_completed()
    robot.camera.color_image_enabled = True


    batch_size = 32
    target_update_steps = batch_size * 4
    n_episodes = 50

    for episode in range(n_episodes):
        done = False
        total_reward = 0
        while not exit_flag and not done:
            input_tensor = get_image(robot)
            #if abs(take_image_and_find_deviation(robot))<125:
            print(f"Step:{step} image passed to model")
            
            # # Move Cozmo based on predicted wheel speeds
            # left_wheel_speed, right_wheel_speed = wheel_speeds[0][0].item(), wheel_speeds[0][1].item()
            #left_probs, right_probs = model(input_tensor)  # Get the probabilities from the model
            left_probs, right_probs = select_action(model, input_tensor.to(device), epsilon)

            left_wheel_speed = torch.argmax(left_probs, dim=1) * 10  # Get the index and convert to value
            right_wheel_speed = torch.argmax(right_probs, dim=1) * 10

            robot.drive_wheels(left_wheel_speed, right_wheel_speed, duration=1)
            #time.sleep(0.5)

            deviation = take_image_and_find_deviation(robot)
            reward = calculate_reward(deviation)
            total_reward += reward
            step += 1
            done = deviation == 1000
            # Store in replay buffer and sample a batch
            replay_buffer.store((input_tensor.squeeze(0), (torch.argmax(left_probs, dim=1), torch.argmax(right_probs, dim=1)), reward, get_image(robot).squeeze(0), done))
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                print("updating model")
                loss = update_dqn(model, target_model, batch,optimizer,device)
                #print(loss)
            if step % target_update_steps == 0:
                print("updating target_model")
                update_target_model(model,target_model)

            if done:
                print("waiting 4 sec...")
                time.sleep(4)
                print(f"EPISODE: {episode+1} TOTAl REWARD: {total_reward}")


            # Save the model
            torch.save(model.state_dict(), model_path)
            #torch.save(target_model.state_dict(),target_model_path)
            save_replay_buffer(replay_buffer,replay_buffer_path)

            # Append training data to CSV
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([input_tensor.squeeze(0).shape, (torch.argmax(left_probs, dim=1), torch.argmax(right_probs, dim=1)), reward, get_image(robot).squeeze(0).shape, done])
                file.close()
            

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Epsilon: {epsilon}")


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)