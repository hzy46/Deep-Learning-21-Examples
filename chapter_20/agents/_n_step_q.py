import os
import time
import numpy as np
import tensorflow as tf
from logging import getLogger

from .agent import Agent
from .history import History
from .experience import Experience

logger = getLogger(__name__)

class NStepQ(Agent):
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

  # Add accumulated gradients for n-step Q-learning
  def make_accumulated_gradients(self):
    reset_accum_grads = []
    new_grads_and_vars = []

    # 1. Prepare accum_grads
    self.accum_grads = {}
    self.add_accum_grads = {}

    for step, network in enumerate(self.networks):
      grads_and_vars = self.global_optim.compute_gradients(network.total_loss, network.w.values())
      _add_accum_grads = []

      for grad, var in tuple(grads_and_vars):
        if grad is not None:
          shape = grad.get_shape().as_list()

          name = 'accum/%s' % "/".join(var.name.split(':')[0].split('/')[-3:])
          if step == 0:
            self.accum_grads[name] = tf.Variable(
                tf.zeros(shape), trainable=False, name=name)

            global_v = global_var[re.sub(r'.*\/A3C_\d+\/', '', var.name)]
            new_grads_and_vars.append((tf.clip_by_norm(self.accum_grads[name].ref(), self.max_grad_norm), global_v))

            reset_accum_grads.append(self.accum_grads[name].assign(tf.zeros(shape)))

          _add_accum_grads.append(tf.assign_add(self.accum_grads[name], grad))

      # 2. Add gradient to accum_grads
      self.add_accum_grads[step] = tf.group(*_add_accum_grads)

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

    # Deep Q-learning
    max_q_t_plus_1 = self.target_network.calc_max_outputs(s_t_plus_1)
    target_q_t = (1. - terminal) * self.discount_r * max_q_t_plus_1 + reward

    _, q_t, loss = self.sess.run([self.optim, self.pred_network.outputs, self.loss], {
      self.targets: target_q_t,
      self.actions: action,
      self.pred_network.inputs: s_t,
    })

    return q_t, loss, True
