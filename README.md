# cozmoproject
## Install the following dependencies:
* pip install cozmo[camera] torch torchvision opencv-python pynput pillow

## In order to use the trained model for inference,simply run inference.py.
(make sure that Cozmo is connected and placed on a proper track/path)
* python inference.py

### cozmo_cam_test.py can be used to see what the preprocessed images look like. 
* python cozmo_cam_test.py

## For retraining the model with or without previous replay_buffer run rl_train.py.
(modify the replay_buffer_path in the rl_train.py file to "new_replay_buffer" to use an empty replay buffer and save it as "new_replay_buffer" later)
* python rl_train.py

## Custom experiences can be added to the replay_buffer using cozmo_drive.py.
running the script cozmo_drive.py will prompt the user to enter right and left wheel speeds for cozmo's step, these experiences will be saved as a new replay buffer which can later be merged with any previous buffer.
* python cozmo_drive.py

### merge_replay.py can be used to merge two replay buffers. 
### view_and_select_replay.py shows the experiences stored in the replay buffer and lets user keep/remove experiences as necessary.
### modify_replay.py can be used to modify the replay buffer experiences (for reward engineering).