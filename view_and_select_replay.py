import torch
import cozmo
import cv2
import numpy as np
import pickle
import os
from collections import deque
import random

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
    # Convert to NumPy array if it's a Tensor
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # Convert to a format suitable for display
    image = np.array(image * 255, dtype=np.uint8)
    if image.shape[0] == 1:  # Single channel image
        image = image.squeeze(0)
    return image

replay_buffer_path = 'replay_buffer' 
new_replay_buffer = ReplayBuffer(100000)  # Initialize new replay buffer

def display_info_on_image(image, left_wheel, right_wheel, reward, done, is_left_image):
    # Set parameters for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 1
    line_type = cv2.LINE_AA

    # Display different information based on whether it's the left or right image
    if is_left_image:
        # Display left_wheel and right_wheel values
        cv2.putText(image, f'Left Wheel: {left_wheel}', (10, 20), font, font_scale, color, thickness, line_type)
        cv2.putText(image, f'Right Wheel: {right_wheel}', (10, 40), font, font_scale, color, thickness, line_type)
    else:
        # Display reward and done values
        cv2.putText(image, f'Reward: {reward}', (10, 20), font, font_scale, color, thickness, line_type)
        cv2.putText(image, f'Done: {done}', (10, 40), font, font_scale, color, thickness, line_type)

    return image

images_dir = 'images'

if os.path.exists(replay_buffer_path):
    replay_buffer = load_replay_buffer(replay_buffer_path)
    print(f"Length of RB: {len(replay_buffer)}")
    step = 0
    for experience in replay_buffer.buffer:
        step += 1
        print(f"Step: {step}")
        # Display the existing and new images from the experience
        existing_image = process_image(experience[0])  # Assuming the existing image is at index 0
        new_image = process_image(experience[3])       # Assuming the new image is at index 3
        left_wheel, right_wheel = experience[1]
        reward = experience[2]
        done = experience[4]

        # Add text to images
        existing_image_with_text = display_info_on_image(existing_image, left_wheel, right_wheel, reward, done, True)
        new_image_with_text = display_info_on_image(new_image, left_wheel, right_wheel, reward, done, False)

        combined_image = np.hstack((existing_image_with_text, new_image_with_text))  # Combine images horizontally
        cv2.imshow('Existing Image (Left) vs New Image (Right)', combined_image)
        # Save the combined image
        image_file = os.path.join(images_dir, f'combined_image_{step}.png')
        cv2.imwrite(image_file, combined_image)
        cv2.waitKey(1)  # Display each set of images for a short time

        print(f"Reward: {experience[2]}")
        user_input = input("Add this experience to new replay buffer? (y/n): ")
        
        if user_input.lower() == 'y':
            new_replay_buffer.store(experience)
    
    cv2.destroyAllWindows()
    #save_replay_buffer(new_replay_buffer, 'new_replay')
    print("New replay buffer saved as 'new_replay'")
