# coding: utf-8

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

# 导入一些需要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# 第一步: 在下面这个地址下载语料库
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
  """
  这个函数的功能是：
      如果filename不存在，就在上面的地址下载它。
      如果filename存在，就跳过下载。
      最终会检查文字的字节数是否和expected_bytes相同。
  """
  if not os.path.exists(filename):
    print('start downloading...')
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

# 下载语料库text8.zip并验证下载
filename = maybe_download('text8.zip', 31344016)



# 将语料库解压，并转换成一个word的list
def read_data(filename):
  """
  这个函数的功能是：
      将下载好的zip文件解压并读取为word的list
  """
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary)) # 总长度为1700万左右
# 输出前100个词。
print(vocabulary[0:100])



# 第二步: 制作一个词表，将不常见的词变成一个UNK标识符
# 词表的大小为5万（即我们只考虑最常出现的5万个词）
vocabulary_size = 50000


def build_dataset(words, n_words):
  """
  函数功能：将原始的单词表示变成index
  """
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # UNK的index为0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # 删除已节省内存
# 输出最常出现的5个单词
print('Most common words (+UNK)', count[:5])
# 输出转换后的数据库data，和原来的单词（前10个）
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# 我们下面就使用data来制作训练集
data_index = 0



# 第三步：定义一个函数，用于生成skip-gram模型用的batch
def generate_batch(batch_size, num_skips, skip_window):
  # data_index相当于一个指针，初始为0
  # 每次生成一个batch，data_index就会相应地往后推
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  # data_index是当前数据开始的位置
  # 产生batch后就往后推1位（产生batch）
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    # 利用buffer生成batch
    # buffer是一个长度为 2 * skip_window + 1长度的word list
    # 一个buffer生成num_skips个数的样本
#     print([reverse_dictionary[i] for i in buffer])
    target = skip_window  # target label at the center of the buffer
#     targets_to_avoid保证样本不重复
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    # 每利用buffer生成num_skips个样本，data_index就向后推进一位
    data_index = (data_index + 1) % len(data)
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

# 默认情况下skip_window=1, num_skips=2
# 此时就是从连续的3(3 = skip_window*2 + 1)个词中生成2(num_skips)个样本。
# 如连续的三个词['used', 'against', 'early']
# 生成两个样本：against -> used, against -> early
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])



# 第四步: 建立模型.

batch_size = 128
embedding_size = 128  # 词嵌入空间是128维的。即word2vec中的vec是一个128维的向量
skip_window = 1       # skip_window参数和之前保持一致
num_skips = 2         # num_skips参数和之前保持一致

# 在训练过程中，会对模型进行验证 
# 验证的方法就是找出和某个词最近的词。
# 只对前valid_window的词进行验证，因为这些词最常出现
valid_size = 16     # 每次验证16个词
valid_window = 100  # 这16个词是在前100个最常见的词中选出来的
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# 构造损失时选取的噪声词的数量
num_sampled = 64

graph = tf.Graph()

with graph.as_default():

  # 输入的batch
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  # 用于验证的词
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # 下面采用的某些函数还没有gpu实现，所以我们只在cpu上定义模型
  with tf.device('/cpu:0'):
    # 定义1个embeddings变量，相当于一行存储一个词的embedding
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 利用embedding_lookup可以轻松得到一个batch内的所有的词嵌入
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # tf.nn.nce_loss会自动选取噪声词，并且形成损失。
  # 随机选取num_sampled个噪声词
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # 得到loss后，我们就可以构造优化器了
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # 计算词和词的相似度（用于验证）
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  # 找出和验证词的embedding并计算它们和所有单词的相似度
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # 变量初始化步骤
  init = tf.global_variables_initializer()



# 第五步：开始训练
num_steps = 100001

with tf.Session(graph=graph) as session:
  # 初始化变量
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # 优化一步
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # 2000个batch的平均损失
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # 每1万步，我们进行一次验证
    if step % 10000 == 0:
      # sim是验证词与所有词之间的相似度
      sim = similarity.eval()
      # 一共有valid_size个验证词
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # 输出最相邻的8个词语
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  # final_embeddings是我们最后得到的embedding向量
  # 它的形状是[vocabulary_size, embedding_size]
  # 每一行就代表着对应index词的词嵌入表示
  final_embeddings = normalized_embeddings.eval()



# Step 6: 可视化
# 可视化的图片会保存为“tsne.png”

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib
  matplotlib.use('agg')
  import matplotlib.pyplot as plt
  # 因为我们的embedding的大小为128维，没有办法直接可视化
  # 所以我们用t-SNE方法进行降维
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  # 只画出500个词的位置
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')

