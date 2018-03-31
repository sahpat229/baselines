import numpy as np
import random

from collections import deque

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.future_y_inputs = RingBuffer(limit, shape=action_shape)
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        future_y_inputs_batch = self.future_y_inputs.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'future_y_inputs': array_min2d(future_y_inputs_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, future_y_input, reward, obs1, terminal1, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)
        self.future_y_inputs.append(future_y_input)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)

class ReplayBufferRollout(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    @property
    def nb_entries(self):
        return len(self.buffer)
    
    def append(self, rollout):
        experience = rollout
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size, gamma):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s1_batch = np.array([exp[0] for exp in batch])
        a1_batch = np.array([exp[1] for exp in batch])
        s1y_batch = np.array([exp[4] for exp in batch])
        sf_batch = np.array([exp[-1] for exp in batch])
        t_batch = np.array([exp[-3] for exp in batch])

        rs_batch = []
        index = 2

        while index < len(batch[0]):
            rs_batch.append(np.array([exp[index] for exp in batch]))
            index += 5

        all_rs = np.zeros(batch_size)
        for r_batch in reversed(rs_batch):
            all_rs *= gamma
            all_rs += r_batch

        result = {
            'obs0': array_min2d(s1_batch),
            'obs1': array_min2d(sf_batch),
            'rewards': array_min2d(all_rs),
            'actions': array_min2d(a1_batch),
            'future_y_inputs': array_min2d(s1y_batch),
            'terminals1': array_min2d(t_batch),
        }

        return result

    def clear(self):
        self.buffer.clear()
        self.count = 0