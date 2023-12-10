import pickle
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

def load_replay_buffer(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_replay_buffer(replay_buffer, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(replay_buffer, f)

def merge_replay_buffers(file_name1, file_name2, output_file_name, capacity):
    buffer1 = load_replay_buffer(file_name1)
    buffer2 = load_replay_buffer(file_name2)

    merged_buffer = ReplayBuffer(capacity)

    for experience in buffer1.buffer:
        merged_buffer.store(experience)
    
    for experience in buffer2.buffer:
        merged_buffer.store(experience)

    save_replay_buffer(merged_buffer, output_file_name)
    print(f"Merged Replay Buffer saved as '{output_file_name}'")

# Example usage
file_name1 = 'new_buffer'
file_name2 = 'replay_buffer'
output_file_name = 'merged_replay_buffer_2'
capacity = 100000  # Set a suitable capacity for the merged buffer

merge_replay_buffers(file_name1, file_name2, output_file_name, capacity)
