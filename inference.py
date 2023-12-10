import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image

import cozmo
import torchvision.transforms as transforms

import time
import os
from pynput import keyboard

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

def get_image(robot, threshold_value=50):
    # Get image from Cozmo's camera
    robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=5)
    pil_initial_image = robot.world.latest_image.raw_image.convert("RGB")
    initial_image_np = np.array(pil_initial_image)
    initial_image_np = initial_image_np[50:initial_image_np.shape[0]-50, 20:280]
    bw_thresh = cv2.cvtColor(initial_image_np, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw_thresh, threshold_value, 255, cv2.THRESH_BINARY_INV)
    input_tensor = transforms.functional.to_tensor(bw).unsqueeze(0)
    # print(input_tensor.shape,input_tensor.unsqueeze(0).shape)
    return input_tensor

def cozmo_program(robot: cozmo.robot.Robot):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'cozmo_model_10'  

    model = CozmoDriveNet().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        model.eval()

    # Set up the listener for the 'q' key to exit
    exit_flag = False
    def on_press(key):
        nonlocal exit_flag
        if key == keyboard.Key.esc or (hasattr(key, 'char') and key.char == 'q'):
            exit_flag = True
            return False  

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    robot.set_head_angle(cozmo.util.degrees(-25)).wait_for_completed()
    robot.set_lift_height(0).wait_for_completed()
    robot.camera.color_image_enabled = True

    while not exit_flag:
        input_tensor = get_image(robot)

        with torch.no_grad():
            left_probs, right_probs = model(input_tensor.to(device))

        left_wheel_speed = torch.argmax(left_probs, dim=1) * 10
        right_wheel_speed = torch.argmax(right_probs, dim=1) * 10

        robot.drive_wheels(left_wheel_speed, right_wheel_speed, duration=1)
        time.sleep(0.2)

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
