import numpy as np
import random
from collections import deque
import multiprocessing as mp

class ReplayBuffer:
    """Buffer to store self-play trajectories.
    """
    def __init__(self, size):
        self._buffer = deque(maxlen=size)

    def sample(self, batch_size):
        batch = random.sample(self._buffer, batch_size)
        states, probs, winners = zip(*batch)
        states = np.stack(states, axis=0)
        probs = np.stack(probs, axis=0)
        winners = np.stack(winners, axis=0)

        return states, probs, winners

    def add(self, item):
        self._buffer.append(item)

    def add_seq(self, seq):
        """Add a trajectory into buffer.

        Args:
            seq (list): List of (board, probs, winner).
        """
        data = self._augment(seq)
        for item in data:
            self.add(item)

    def _augment(self, data):
        """Augment model input by rotating and flipping.

        Args:
            data (list): List of (model_input, probs, winner)

        Returns:
            list: List of (model_input, probs, winner)
        """
        augmented_data = []
        for feat, probs, winner in data:
            for k in range(4):
                rot_feat = np.rot90(feat, k, axes=(-2, -1))
                # print("feat\n", feat, f"rot{k}\n", rot_feat)
                rot_probs = np.rot90(probs.reshape(*feat.shape[-2:]), k, axes=(-2, -1))
                # print("probs\n", probs, f"rot{k}\n", rot_probs)
                augmented_data.append((rot_feat, rot_probs.flatten(), winner))
                augmented_data.append((rot_feat[..., ::-1], rot_probs[..., ::-1].flatten(), winner))
        return augmented_data

    def __len__(self):
        return len(self._buffer)
    
class SharedReplayBuffer:
    """Buffer to store self-play trajectories.
    """
    def __init__(self, size):
        self._buffer = SharedQueue(maxlen=size)

    def sample(self, batch_size):
        batch = random.sample(list(self._buffer), batch_size)
        states, probs, winners = zip(*batch)
        states = np.stack(states, axis=0)
        probs = np.stack(probs, axis=0)
        winners = np.stack(winners, axis=0)

        return states, probs, winners

    def add(self, item):
        self._buffer.put(item)

    def add_seq(self, seq):
        """Add a trajectory into buffer.

        Args:
            seq (list): List of (board, probs, winner).
        """
        data = self._augment(seq)
        for item in data:
            self.add(item)

    def _augment(self, data):
        """Augment model input by rotating and flipping.

        Args:
            data (list): List of (model_input, probs, winner)

        Returns:
            list: List of (model_input, probs, winner)
        """
        augmented_data = []
        for feat, probs, winner in data:
            for k in range(4):
                rot_feat = np.rot90(feat, k, axes=(-2, -1))
                # print("feat\n", feat, f"rot{k}\n", rot_feat)
                rot_probs = np.rot90(probs.reshape(*feat.shape[-2:]), k, axes=(-2, -1))
                # print("probs\n", probs, f"rot{k}\n", rot_probs)
                augmented_data.append((rot_feat, rot_probs.flatten(), winner))
                augmented_data.append((rot_feat[..., ::-1], rot_probs[..., ::-1].flatten(), winner))
        return augmented_data

    def __len__(self):
        return len(self._buffer)
    
class SharedQueue:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.queue = mp.Manager().list()
        self.lock = mp.Lock()

    def put(self, item):
        with self.lock:
            while len(self.queue) >= self.maxlen:
                self.queue.pop(0)
            self.queue.append(item)

    def empty(self):
        return len(self.queue) == 0

    def __iter__(self):
        return iter(self.queue)
    
    def __len__(self):
        return len(self.queue)