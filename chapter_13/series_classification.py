# coding: utf-8
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np

# 这个类用于产生序列样本
class ToySequenceData(object):
    """ 生成序列数据。每个数量可能具有不同的长度。
    一共生成下面两类数据
    - 类别 0: 线性序列 (如 [0, 1, 2, 3,...])
    - 类别 1: 完全随机的序列 (i.e. [1, 3, 10, 7,...])
    注意:
    max_seq_len是最大的序列长度。对于长度小于这个数值的序列，我们将会补0。
    在送入RNN计算时，会借助sequence_length这个属性来进行相应长度的计算。
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # 序列的长度是随机的，在min_seq_len和max_seq_len之间。
            len = random.randint(min_seq_len, max_seq_len)
            # self.seqlen用于存储所有的序列。
            self.seqlen.append(len)
            # 以50%的概率，随机添加一个线性或随机的训练
            if random.random() < .5:
                # 生成一个线性序列
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # 长度不足max_seq_len的需要补0
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                # 线性序列的label是[1, 0]（因为我们一共只有两类）
                self.labels.append([1., 0.])
            else:
                # 生成一个随机序列
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # 长度不足max_seq_len的需要补0
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        """
        生成batch_size的样本。
        如果使用完了所有样本，会重新从头开始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen



# 这一部分只是测试一下如何使用上面定义的ToySequenceData
tmp = ToySequenceData()

# 生成样本
batch_data, batch_labels, batch_seqlen = tmp.next(32)

# batch_data是序列数据，它是一个嵌套的list，形状为(batch_size, max_seq_len, 1)
print(np.array(batch_data).shape)  # (32, 20, 1)

# 我们之前调用tmp.next(32)，因此一共有32个序列
# 我们可以打出第一个序列
print(batch_data[0])

# batch_labels是label，它也是一个嵌套的list，形状为(batch_size, 2)
# (batch_size, 2)中的“2”表示为两类分类
print(np.array(batch_labels).shape)  # (32, 2)

# 我们可以打出第一个序列的label
print(batch_labels[0])

# batch_seqlen一个长度为batch_size的list，表示每个序列的实际长度
print(np.array(batch_seqlen).shape)  # (32,)

# 我们可以打出第一个序列的长度
print(batch_seqlen[0])




# 运行的参数
learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 10

# 网络定义时的参数
seq_max_len = 20 # 最大的序列长度
n_hidden = 64 # 隐层的size
n_classes = 2 # 类别数

trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# x为输入，y为输出
# None的位置实际为batch_size
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# 这个placeholder存储了输入的x中，每个序列的实际长度
seqlen = tf.placeholder(tf.int32, [None])

# weights和bias在输出时会用到
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):

    # 输入x的形状： (batch_size, max_seq_len, n_input)
    # 输入seqlen的形状：(batch_size, )
    

    # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # 使用tf.nn.dynamic_rnn展开时间维度
    # 此外sequence_length=seqlen也很重要，它告诉TensorFlow每一个序列应该运行多少步
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    
    # outputs的形状为(batch_size, max_seq_len, n_hidden)
    # 如果有疑问可以参考上一章内容

    # 我们希望的是取出与序列长度相对应的输出。如一个序列长度为10，我们就应该取出第10个输出
    # 但是TensorFlow不支持直接对outputs进行索引，因此我们用下面的方法来做：

    batch_size = tf.shape(outputs)[0]
    # 得到每一个序列真正的index
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # 给最后的输出
    return tf.matmul(outputs, weights['out']) + biases['out']

# 这里的pred是logits而不是概率
pred = dynamicRNN(x, seqlen, weights, biases)

# 因为pred是logits，因此用tf.nn.softmax_cross_entropy_with_logits来定义损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 分类准确率
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # 每run一次就会更新一次参数
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # 在这个batch内计算准确度
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # 在这个batch内计算损失
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # 最终，我们在测试集上计算一次准确度
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
