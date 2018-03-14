#coding: utf-8
# 导入当前目录的cifar10_input，这个模块负责读入cifar10数据
import cifar10_input
# 导入TensorFlow和其他一些可能用到的模块。
import tensorflow as tf
import os
import scipy.misc


def inputs_origin(data_dir):
  # filenames一共5个，从data_batch_1.bin到data_batch_5.bin
  # 读入的都是训练图像
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  # 判断文件是否存在
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  # 将文件名的list包装成TensorFlow中queue的形式
  filename_queue = tf.train.string_input_producer(filenames)
  # cifar10_input.read_cifar10是事先写好的从queue中读取文件的函数
  # 返回的结果read_input的属性uint8image就是图像的Tensor
  read_input = cifar10_input.read_cifar10(filename_queue)
  # 将图片转换为实数形式
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  # 返回的reshaped_image是一张图片的tensor
  # 我们应当这样理解reshaped_image：每次使用sess.run(reshaped_image)，就会取出一张图片
  return reshaped_image

if __name__ == '__main__':
  # 创建一个会话sess
  with tf.Session() as sess:
    # 调用inputs_origin。cifar10_data/cifar-10-batches-bin是我们下载的数据的文件夹位置
    reshaped_image = inputs_origin('cifar10_data/cifar-10-batches-bin')
    # 这一步start_queue_runner很重要。
    # 我们之前有filename_queue = tf.train.string_input_producer(filenames)
    # 这个queue必须通过start_queue_runners才能启动
    # 缺少start_queue_runners程序将不能执行
    threads = tf.train.start_queue_runners(sess=sess)
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 创建文件夹cifar10_data/raw/
    if not os.path.exists('cifar10_data/raw/'):
      os.makedirs('cifar10_data/raw/')
    # 保存30张图片
    for i in range(30):
      # 每次sess.run(reshaped_image)，都会取出一张图片
      image_array = sess.run(reshaped_image)
      # 将图片保存
      scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.jpg' % i)
