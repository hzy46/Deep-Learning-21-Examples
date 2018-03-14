"""Modification of https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import random
import numpy as np

class Experience(object):
  def __init__(self, data_format, batch_size, history_length, memory_size, observation_dims):
    self.data_format = data_format
    self.batch_size = batch_size
    self.history_length = history_length
    self.memory_size = memory_size

    self.actions = np.empty(self.memory_size, dtype=np.uint8)
    self.rewards = np.empty(self.memory_size, dtype=np.int8)
    self.observations = np.empty([self.memory_size] + observation_dims, dtype=np.uint8)
    self.terminals = np.empty(self.memory_size, dtype=np.bool)

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty([self.batch_size, self.history_length] + observation_dims, dtype = np.float16)
    self.poststates = np.empty([self.batch_size, self.history_length] + observation_dims, dtype = np.float16)

    self.count = 0
    self.current = 0

  def add(self, observation, reward, action, terminal):
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.observations[self.current, ...] = observation
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def sample(self):
    indexes = []
    while len(indexes) < self.batch_size:
      while True:
        index = random.randint(self.history_length, self.count - 1)
        if index >= self.current and index - self.history_length < self.current:
          continue
        if self.terminals[(index - self.history_length):index].any():
          continue
        break
      
      self.prestates[len(indexes), ...] = self.retreive(index - 1)
      self.poststates[len(indexes), ...] = self.retreive(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.data_format == 'NHWC' and len(self.prestates.shape) == 4:
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def retreive(self, index):
    index = index % self.count
    if index >= self.history_length - 1:
      return self.observations[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.observations[indexes, ...]
