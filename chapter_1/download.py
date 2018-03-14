# coding:utf-8
# 从tensorflow.examples.tutorials.mnist引入模块。这是TensorFlow为了教学MNIST而提前编制的程序
from tensorflow.examples.tutorials.mnist import input_data
# 从MNIST_data/中读取MNIST数据。这条语句在数据不存在时，会自动执行下载
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 查看训练数据的大小
print(mnist.train.images.shape)  # (55000, 784)
print(mnist.train.labels.shape)  # (55000, 10)

# 查看验证数据的大小
print(mnist.validation.images.shape)  # (5000, 784)
print(mnist.validation.labels.shape)  # (5000, 10)

# 查看测试数据的大小
print(mnist.test.images.shape)  # (10000, 784)
print(mnist.test.labels.shape)  # (10000, 10)

# 打印出第0幅图片的向量表示
print(mnist.train.images[0, :])

# 打印出第0幅图片的标签
print(mnist.train.labels[0, :])
