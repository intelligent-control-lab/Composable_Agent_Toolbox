import random
from collections import deque, namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(
        self,
        batch_size,
        buffer_size
    ):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=self.buffer_size)

    def __len__(self):
        return len(self.replay_buffer)

    def sample(self):
        if self.batch_size > len(self.replay_buffer): return None
        return random.sample(self.replay_buffer, self.batch_size)

    def append(self, *args):
        self.replay_buffer.append(Transition(*args))
