import numpy as np

class History:
  def __init__(self, data_format, batch_size, history_length, screen_dims):
    self.data_format = data_format
    self.history = np.zeros([history_length] + screen_dims, dtype=np.float32)

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    if self.data_format == 'NHWC' and len(self.history.shape) == 3:
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history
