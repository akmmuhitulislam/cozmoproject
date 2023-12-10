# cozmoproject
## Install the following dependencies:
pip install cozmo[camera]
pip install torch torchvision
pip install opencv-python

## In order to use the trained model for inference,simply run inference.py.
(make sure that Cozmo is connected and placed on a proper track/path)
python inference.py

## For retraining the model with or without previous replay_buffer run rl_train.py.
(modify the replay_buffer_path in the rl_train.py file to "" to use an empty replay buffer)
python rl_train.py

## cozmo_cam_test.py can be used to see what the preprocessed images look like. 

## merge_replay.py can be used to merge two replay buffers. 
## view_and_select_replay.py shows the experiences stored in the replay buffer and lets user keep/remove experiences as necessary.
## modify_replay.py can be used to modify the replay buffer experiences (for reward engineering).