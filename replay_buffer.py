import random
from collections import deque

from transition import Transition


class ReplayBuffer(object):
    """
    Stores environment transitions for experience replay.

    The replay buffer enables off-policy learning by storing past
    transitions and sampling random minibatches during training.
    Random sampling helps reduce temporal correlation between
    experiences and improves training stability.
    """

    # Experience Replay Buffer (off-policy data storage)
    def __init__(self, capacity):
        """
        Initializes the replay buffer with a fixed maximum capacity.

        Args:
            capacity (int):
                Maximum number of transitions stored in memory.
                Older transitions are discarded once capacity is reached.
        """
        self.memory = deque([], maxlen=capacity)

    def enqueue(self, *args):
        """
        Stores a transition in the replay buffer.

        Args:
            *args:
                Components of a transition tuple:
                (state, action, next_state, reward
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly samples a minibatch of transitions from memory.

        Args:
            batch_size (int):
                 Number of transitions to sample.

        Returns:
            list (Transition):
                Random minibatch of stored transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current number of stored transitions.

        Returns:
            int:
                Number of transitions currently stored in memory.
        """
        return len(self.memory)
