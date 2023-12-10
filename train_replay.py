import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
import cozmo
import cv2
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming CozmoDriveNet and ReplayBuffer are already defined

# Initialize the policy network (the main network being trained)
policy_net = CozmoDriveNet().to(device)
model_path = "cozmo_model_10"
if os.path.exists(model_path):
    policy_net.load_state_dict(torch.load(model_path))

# Initialize the target network and set its weights to policy_net's weights
# The target network's weights are kept frozen most of the time
target_net = CozmoDriveNet().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set the target network to evaluation mode

# Define the optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# Load or initialize your ReplayBuffer
replay_buffer_path = 'replay_buffer'

#replay_buffer = ReplayBuffer(100000)
if os.path.exists(replay_buffer_path):
    replay_buffer = load_replay_buffer(replay_buffer_path)
    print(len(replay_buffer))

# Training parameters
batch_size = 32
gamma = 0.95  # Discount factor for future rewards
num_epochs = 100
update_target_every = batch_size * 4  # How often to update the target network

# Function to convert tensor to numpy and vice versa
def to_tensor(np_array, device=torch.device('cpu')):
    return torch.from_numpy(np_array).float().to(device)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

# Training loop
for epoch in range(num_epochs):
    for i in range(batch_size, len(replay_buffer)):
        start_index = i - batch_size
        end_index = start_index + batch_size
        experiences = [replay_buffer.buffer[j] for j in range(start_index, end_index)]
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device) 
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        # Current Q values are estimated by the model for all actions
        current_q_probs_left, current_q_probs_right = policy_net(states)
        current_q_values_left = current_q_probs_left.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1)
        current_q_values_right = current_q_probs_right.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1)

        # Next Q values are estimated by the model's target network
        next_q_probs_left, next_q_probs_right = target_net(next_states)
        max_next_q_values_left = next_q_probs_left.detach().max(1)[0]
        max_next_q_values_right = next_q_probs_right.detach().max(1)[0]

        # target Q values
        target_q_values_left = rewards + (gamma * max_next_q_values_left)
        target_q_values_right = rewards + (gamma * max_next_q_values_right)

        # Compute the loss
        loss_left = F.mse_loss(current_q_values_left, target_q_values_left)
        loss_right = F.mse_loss(current_q_values_right, target_q_values_right)
        loss = loss_left + loss_right

        # Zero gradients, backward pass, optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update the target network
    if epoch % update_target_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained policy network
# torch.save(policy_net.state_dict(), 'cozmo_net_dqn_trained.pth')
torch.save(policy_net.state_dict(), 'cozmo_model_11')