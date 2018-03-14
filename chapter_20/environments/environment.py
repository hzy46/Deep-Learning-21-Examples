import gym
import random
import logging
import numpy as np

from .corridor import CorridorEnv

try:
  import scipy.misc
  imresize = scipy.misc.imresize
  imwrite = scipy.misc.imsave
except:
  import cv2
  imresize = cv2.resize
  imwrite = cv2.imwrite

logger = logging.getLogger(__name__)

class Environment(object):
  def __init__(self, env_name, n_action_repeat, max_random_start,
               observation_dims, data_format, display, use_cumulated_reward=False):
    self.env = gym.make(env_name)

    self.n_action_repeat = n_action_repeat
    self.max_random_start = max_random_start
    self.action_size = self.env.action_space.n

    self.display = display
    self.data_format = data_format
    self.observation_dims = observation_dims
    self.use_cumulated_reward = use_cumulated_reward

    if hasattr(self.env, 'get_action_meanings'):
      logger.info("Using %d actions : %s" % (self.action_size, ", ".join(self.env.get_action_meanings())))

  def new_game(self):
    return self.preprocess(self.env.reset()), 0, False

  def new_random_game(self):
    return self.new_game()

  def step(self, action, is_training=False):
    observation, reward, terminal, info = self.env.step(action)
    if self.display: self.env.render()
    return self.preprocess(observation), reward, terminal, info

  def preprocess(self):
    raise NotImplementedError()

class ToyEnvironment(Environment):
  def preprocess(self, obs):
    new_obs = np.zeros([self.env.observation_space.n])
    new_obs[obs] = 1
    return new_obs

class AtariEnvironment(Environment):
  def __init__(self, env_name, n_action_repeat, max_random_start,
               observation_dims, data_format, display, use_cumulated_reward):
    super(AtariEnvironment, self).__init__(env_name, 
        n_action_repeat, max_random_start, observation_dims, data_format, display, use_cumulated_reward)

  def new_game(self, from_random_game=False):
    screen = self.env.reset()
    screen, reward, terminal, _ = self.env.step(0)

    if self.display:
      self.env.render()

    if from_random_game:
      return screen, 0, False
    else:
      self.lives = self.env.unwrapped.ale.lives()
      terminal = False
      return self.preprocess(screen, terminal), 0, terminal

  def new_random_game(self):
    screen, reward, terminal = self.new_game(True)

    for idx in range(random.randrange(self.max_random_start)):
      screen, reward, terminal, _ = self.env.step(0)

      if terminal: logger.warning("warning: terminal signal received after %d 0-steps", idx)

    if self.display:
      self.env.render()

    self.lives = self.env.unwrapped.ale.lives()

    terminal = False
    return self.preprocess(screen, terminal), 0, terminal

  def step(self, action, is_training):
    if action == -1:
      # Step with random action
      action = self.env.action_space.sample()

    cumulated_reward = 0

    for _ in range(self.n_action_repeat):
      screen, reward, terminal, _ = self.env.step(action)
      cumulated_reward += reward
      current_lives = self.env.unwrapped.ale.lives()

      if is_training and self.lives > current_lives:
        terminal = True

      if terminal: break

    if self.display:
      self.env.render()

    if not terminal:
      self.lives = current_lives

    if self.use_cumulated_reward:
      return self.preprocess(screen, terminal), cumulated_reward, terminal, {}
    else:
      return self.preprocess(screen, terminal), reward, terminal, {}

  def preprocess(self, raw_screen, terminal):
    y = 0.2126 * raw_screen[:, :, 0] + 0.7152 * raw_screen[:, :, 1] + 0.0722 * raw_screen[:, :, 2]
    y = y.astype(np.uint8)
    y_screen = imresize(y, self.observation_dims)
    return y_screen
