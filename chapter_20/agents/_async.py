import os
import time
import numpy as np
import tensorflow as tf
from threading import Thread
from logging import getLogger

from .agent import Agent
from .history import History
from .experience import Experience

logger = getLogger(__name__)

class Async(Agent):
  def __init__(self, sess, pred_network, env, stat, conf, target_network=None):
    super(DeepQ, self).__init__(sess, pred_network, target_network, env, stat, conf)

    raise Exception("[!] Not fully implemented yet")

    # Optimizer
    with tf.variable_scope('optimizer'):
      self.targets = tf.placeholder('float32', [None], name='target_q_t')
      self.actions = tf.placeholder('int64', [None], name='action')

      actions_one_hot = tf.one_hot(self.actions, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      pred_q = tf.reduce_sum(self.pred_network.outputs * actions_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.targets - pred_q
      if self.max_delta and self.min_delta:
        self.delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

      self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')

      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.stat.t_op,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))

      optimizer = tf.train.RMSPropOptimizer(
        self.learning_rate_op, momentum=0.95, epsilon=0.01)
      
      grads_and_vars = optimizer.compute_gradients(self.loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars)


  def train(self, max_t, global_t):
    self.global_t = global_t

    # 0. Prepare training
    state, reward, terminal = self.env.new_random_game()
    self.observe(state, reward, terminal)

    while True:
      if global_t[0] > self.t_train_max:
        break

      # 1. Predict
      action = self.predict(state)
      # 2. Step
      state, reward, terminal = self.env.step(action, is_training=True)
      # 3. Observe
      self.observe(state, reward, terminal)

      if terminal:
        observation, reward, terminal = self.new_game()

      global_t[0] += 1

  def train_with_log(self, max_t, global_t):
    from tqdm import tqdm

    for _ in tqdm(range(max_t), ncols=70, initial=int(global_t[0])):
      if global_t[0] > self.t_train_max:
        break

      # 1. Predict
      action = self.predict(state)
      # 2. Step
      state, reward, terminal = self.env.step(-1, is_training=True)
      # 3. Observe
      self.observe(state, reward, terminal)

      if terminal:
        observation, reward, terminal = self.new_game()

      global_t[0] += 1

    if self.stat:
      self.stat.on_step(self.t, action, reward, terminal,
                        ep, q, loss, is_update, self.learning_rate_op)

  def observe(self, s_t, r_t, terminal):
    self.prev_r[self.t] = max(self.min_reward, min(self.max_reward, r_t))

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      r = {}

      lr = (self.t_train_max - self.global_t[0] + 1) / self.t_train_max * self.learning_rate

      if terminal:
        r[self.t] = 0.
      else:
        r[self.t] = self.sess.partial_run(
            self.partial_graph,
            self.networks[self.t_start - self.t].value,
        )[0][0]

      for t in range(self.t - 1, self.t_start - 1, -1):
        r[t] = self.prev_r[t] + self.gamma * r[t + 1]

      data = {}
      data.update({
        self.networks[t].R: [r[t + self.t_start]] for t in range(len(self.prev_r) - 1)
      })
      data.update({
        self.networks[t].true_log_policy:
          [self.prev_log_policy[t + self.t_start]] for t in range(len(self.prev_r) - 1)
      })
      data.update({
        self.learning_rate_op: lr,
      })

      # 1. Update accumulated gradients
      if not self.writer:
        self.sess.partial_run(self.partial_graph,
            [self.add_accum_grads[t] for t in range(len(self.prev_r) - 1)], data)
      else:
        results = self.sess.partial_run(self.partial_graph,
            [self.value_policy_summary] + [self.add_accum_grads[t] for t in range(len(self.prev_r) - 1)], data)

        summary_str = results[0]
        self.writer.add_summary(summary_str, self.global_t[0])

      # 2. Update global w with accumulated gradients
      self.sess.run(self.apply_gradient, {
        self.learning_rate_op: lr,
      })

      # 3. Reset accumulated gradients to zero
      self.sess.run(self.reset_accum_grad)

      # 4. Copy weights of global_network to local_network
      self.networks[0].copy_from_global()

      self.prev_r = {self.t: self.prev_r[self.t]}
      self.t_start = self.t

      del self.partial_graph
