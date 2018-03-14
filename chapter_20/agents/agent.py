import gym
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from logging import getLogger

from .history import History
from .experience import Experience

logger = getLogger(__name__)

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

class Agent(object):
  def __init__(self, sess, pred_network, env, stat, conf, target_network=None):
    self.sess = sess
    self.stat = stat

    self.ep_start = conf.ep_start
    self.ep_end = conf.ep_end
    self.history_length = conf.history_length
    self.t_ep_end = conf.t_ep_end
    self.t_learn_start = conf.t_learn_start
    self.t_train_freq = conf.t_train_freq
    self.t_target_q_update_freq = conf.t_target_q_update_freq
    self.env_name = conf.env_name

    self.discount_r = conf.discount_r
    self.min_r = conf.min_r
    self.max_r = conf.max_r
    self.min_delta = conf.min_delta
    self.max_delta = conf.max_delta
    self.max_grad_norm = conf.max_grad_norm
    self.observation_dims = conf.observation_dims

    self.learning_rate = conf.learning_rate
    self.learning_rate_minimum = conf.learning_rate_minimum
    self.learning_rate_decay = conf.learning_rate_decay
    self.learning_rate_decay_step = conf.learning_rate_decay_step

    # network
    self.double_q = conf.double_q
    self.pred_network = pred_network
    self.target_network = target_network
    self.target_network.create_copy_op(self.pred_network)

    self.env = env 
    self.history = History(conf.data_format,
        conf.batch_size, conf.history_length, conf.observation_dims)
    self.experience = Experience(conf.data_format,
        conf.batch_size, conf.history_length, conf.memory_size, conf.observation_dims)

    if conf.random_start:
      self.new_game = self.env.new_random_game
    else:
      self.new_game = self.env.new_game

  def train(self, t_max):
    tf.global_variables_initializer().run()

    self.stat.load_model()
    self.target_network.run_copy()

    start_t = self.stat.get_t()
    observation, reward, terminal = self.new_game()

    for _ in range(self.history_length):
      self.history.add(observation)

    for self.t in tqdm(range(start_t, t_max), ncols=70, initial=start_t):
      ep = (self.ep_end +
          max(0., (self.ep_start - self.ep_end)
            * (self.t_ep_end - max(0., self.t - self.t_learn_start)) / self.t_ep_end))

      # 1. predict
      action = self.predict(self.history.get(), ep)
      # 2. act
      observation, reward, terminal, info = self.env.step(action, is_training=True)
      # 3. observe
      q, loss, is_update = self.observe(observation, reward, action, terminal)

      logger.debug("a: %d, r: %d, t: %d, q: %.4f, l: %.2f" % \
          (action, reward, terminal, np.mean(q), loss))

      if self.stat:
        self.stat.on_step(self.t, action, reward, terminal,
                          ep, q, loss, is_update, self.learning_rate_op)
      if terminal:
        observation, reward, terminal = self.new_game()

  def play(self, test_ep, n_step=10000, n_episode=100):
    tf.initialize_all_variables().run()

    self.stat.load_model()
    self.target_network.run_copy()

    if not self.env.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      env = gym.wrappers.Monitor(self.env.env, gym_dir)

    best_reward, best_idx, best_count = 0, 0, 0
    try:
      itr = xrange(n_episode)
    except NameError:
      itr = range(n_episode)
    for idx in itr:
      observation, reward, terminal = self.new_game()
      current_reward = 0

      for _ in range(self.history_length):
        self.history.add(observation)

      for self.t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(self.history.get(), test_ep)
        # 2. act
        observation, reward, terminal, info = self.env.step(action, is_training=False)
        # 3. observe
        q, loss, is_update = self.observe(observation, reward, action, terminal)

        logger.debug("a: %d, r: %d, t: %d, q: %.4f, l: %.2f" % \
            (action, reward, terminal, np.mean(q), loss))
        current_reward += reward

        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx
        best_count = 0
      elif current_reward == best_reward:
        best_count += 1

      print ("="*30)
      print (" [%d] Best reward : %d (dup-percent: %d/%d)" % (best_idx, best_reward, best_count, n_episode))
      print ("="*30)

    #if not self.env.display:
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')

  def predict(self, s_t, ep):
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.pred_network.calc_actions([s_t])[0]
    return action

  def q_learning_minibatch_test(self):
    s_t = np.array([[[ 0., 0., 0., 0.],
                     [ 0., 0., 0., 0.],
                     [ 0., 0., 0., 0.],
                     [ 1., 0., 0., 0.]]], dtype=np.uint8)
    s_t_plus_1 = np.array([[[ 0., 0., 0., 0.],
                            [ 0., 0., 0., 0.],
                            [ 1., 0., 0., 0.],
                            [ 0., 0., 0., 0.]]], dtype=np.uint8)
    s_t = s_t.reshape([1, 1] + self.observation_dims)
    s_t_plus_1 = s_t_plus_1.reshape([1, 1] + self.observation_dims)

    action = [3]
    reward = [1]
    terminal = [0]

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = self.target_network.calc_max_outputs(s_t_plus_1)
    target_q_t = (1. - terminal) * self.discount_r * max_q_t_plus_1 + reward

    _, q_t, a, loss = self.sess.run([
        self.optim, self.pred_network.outputs, self.pred_network.actions, self.loss
      ], {
        self.targets: target_q_t,
        self.actions: action,
        self.pred_network.inputs: s_t,
      })

    logger.info("q: %s, a: %d, l: %.2f" % (q_t, a, loss))

  def update_target_q_network(self):
    assert self.target_network != None
    self.target_network.run_copy()
