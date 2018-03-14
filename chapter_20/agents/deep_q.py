import os
import time
import numpy as np
import tensorflow as tf
from logging import getLogger

from .agent import Agent

logger = getLogger(__name__)

class DeepQ(Agent):
  def __init__(self, sess, pred_network, env, stat, conf, target_network=None):
    super(DeepQ, self).__init__(sess, pred_network, env, stat, conf, target_network=target_network)

    # Optimizer
    with tf.variable_scope('optimizer'):
      self.targets = tf.placeholder('float32', [None], name='target_q_t')
      self.actions = tf.placeholder('int64', [None], name='action')

      actions_one_hot = tf.one_hot(self.actions, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      pred_q = tf.reduce_sum(self.pred_network.outputs * actions_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.targets - pred_q
      self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
                                    0.5 * tf.square(self.delta),
                                    tf.abs(self.delta) - 0.5, name='clipped_error')

      self.loss = tf.reduce_mean(self.clipped_error, name='loss')

      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.stat.t_op,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))

      optimizer = tf.train.RMSPropOptimizer(
        self.learning_rate_op, momentum=0.95, epsilon=0.01)
      
      if self.max_grad_norm != None:
        grads_and_vars = optimizer.compute_gradients(self.loss)
        for idx, (grad, var) in enumerate(grads_and_vars):
          if grad is not None:
            grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
        self.optim = optimizer.apply_gradients(grads_and_vars)
      else:
        self.optim = optimizer.minimize(self.loss)

  def observe(self, observation, reward, action, terminal):
    reward = max(self.min_r, min(self.max_r, reward))

    self.history.add(observation)
    self.experience.add(observation, reward, action, terminal)

    # q, loss, is_update
    result = [], 0, False

    if self.t > self.t_learn_start:
      if self.t % self.t_train_freq == 0:
        result = self.q_learning_minibatch()

      if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
        self.update_target_q_network()

    return result

  def q_learning_minibatch(self):
    if self.experience.count < self.history_length:
      return [], 0, False
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.experience.sample()

    terminal = np.array(terminal) + 0.

    if self.double_q:
      # Double Q-learning
      pred_action = self.pred_network.calc_actions(s_t_plus_1)
      q_t_plus_1_with_pred_action = self.target_network.calc_outputs_with_idx(
          s_t_plus_1, [[idx, pred_a] for idx, pred_a in enumerate(pred_action)])
      target_q_t = (1. - terminal) * self.discount_r * q_t_plus_1_with_pred_action + reward
    else:
      # Deep Q-learning
      max_q_t_plus_1 = self.target_network.calc_max_outputs(s_t_plus_1)
      target_q_t = (1. - terminal) * self.discount_r * max_q_t_plus_1 + reward

    _, q_t, loss = self.sess.run([self.optim, self.pred_network.outputs, self.loss], {
      self.targets: target_q_t,
      self.actions: action,
      self.pred_network.inputs: s_t,
    })

    return q_t, loss, True
