# coding:utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym

# gym环境
env = gym.make('CartPole-v0')

# 超参数
D = 4  # 输入层神经元个数
H = 10  # 隐层神经元个数
batch_size = 5  # 一个batch中有5个episode，即5次游戏
learning_rate = 1e-2  # 学习率
gamma = 0.99  # 奖励折扣率gamma


# 定义policy网络
# 输入观察值，输出右移的概率
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# 定义和训练、loss有关的变量
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# 定义loss函数
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# 优化器、梯度。
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


def discount_rewards(r):
    """
    输入：
        1维的float类型数组，表示每个时刻的奖励
    输出：
        计算折扣率gamma后的期望奖励
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    # observation是环境的初始观察量（输入神经网络的值）
    observation = env.reset()

    # gradBuffer会存储梯度，此处做一初始化
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        # 当一个batch内的平均奖励达到180以上时，显示游戏窗口
        if reward_sum / batch_size > 180 or rendering is True:
            env.render()
            rendering = True

        # 输入神经网络的值
        x = np.reshape(observation, [1, D])

        # action=1表示向右移
        # action=0表示向左移
        # tfprob为网络输出的向右走的概率
        tfprob = sess.run(probability, feed_dict={observations: x})
        # np.random.uniform()为0~1之间的随机数
        # 当它小于tfprob时，就采取右移策略，反之左移
        action = 1 if np.random.uniform() < tfprob else 0

        # xs记录每一步的观察量，ys记录每一步采取的策略
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        # 执行action
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # drs记录每一步的reward
        drs.append(reward)

        # 一局游戏结束
        if done:
            episode_number += 1
            # 将xs、ys、drs从list变成numpy数组形式
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

            # 对epr计算期望奖励
            discounted_epr = discount_rewards(epr)
            # 对期望奖励做归一化
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr //= np.std(discounted_epr)

            # 将梯度存到gradBuffer中
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # 每batch_size局游戏，就将gradBuffer中的梯度真正更新到policy网络中
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # 打印一些信息
                print('Episode: %d ~ %d Average reward: %f.  ' % (episode_number - batch_size + 1, episode_number, reward_sum // batch_size))

                # 当我们在batch_size游戏中平均能拿到200的奖励，就停止训练
                if reward_sum // batch_size >= 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(episode_number, 'Episodes completed.')
