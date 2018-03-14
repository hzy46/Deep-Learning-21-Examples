import tensorflow as tf
from functools import reduce
from tensorflow.contrib.layers.python.layers import initializers

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d',
           trainable=True):
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, 
        tf.float32, initializer=weights_initializer, trainable=trainable)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

    b = tf.get_variable('b', [output_dim],
        tf.float32, initializer=biases_initializer, trainable=trainable)
    out = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b

def linear(input_,
           output_size,
           weights_initializer=initializers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=None,
           trainable=True,
           name='linear'):
  shape = input_.get_shape().as_list()

  if len(shape) > 2:
    input_ = tf.reshape(input_, [-1, reduce(lambda x, y: x * y, shape[1:])])
    shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('w', [shape[1], output_size], tf.float32,
        initializer=weights_initializer, trainable=trainable)
    b = tf.get_variable('b', [output_size],
        initializer=biases_initializer, trainable=trainable)
    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b

def batch_sample(probs, name='batch_sample'):
  with tf.variable_scope(name):
    uniform = tf.random_uniform(tf.shape(probs), minval=0, maxval=1)
    samples = tf.argmax(probs - uniform, dimension=1)
  return samples
