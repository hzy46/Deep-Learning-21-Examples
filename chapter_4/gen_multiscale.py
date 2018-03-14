# coding:utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf


graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img / 255 * (max - min) + min
    return img


def calc_grad_tiled(img, t_grad, tile_size=512):
    # 每次只对tile_size×tile_size大小的图像计算梯度，避免内存问题
    sz = tile_size
    h, w = img.shape[:2]
    # img_shift：先在行上做整体移动，再在列上做整体移动
    # 防止在tile的边缘产生边缘效应
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    # y, x是开始位置的像素
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            # 每次对sub计算梯度。sub的大小是tile_size×tile_size
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    # 使用np.roll移动回去
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_multiscale(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    # 同样定义目标和梯度
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            # 每次将将图片放大octave_scale倍
            # 共放大octave_n - 1 次
            img = resize_ratio(img, octave_scale)
        for i in range(iter_n):
            # 调用calc_grad_tiled计算任意大小图像的梯度
            g = calc_grad_tiled(img, t_grad)
            g /= g.std() + 1e-8
            img += g * step
            print('.', end=' ')
    savearray(img, 'multiscale.jpg')

if __name__ == '__main__':
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
    render_multiscale(layer_output[:, :, :, channel], img_noise, iter_n=20)
