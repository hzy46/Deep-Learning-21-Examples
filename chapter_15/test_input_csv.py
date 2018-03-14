# coding: utf-8
from __future__ import print_function
import tensorflow as tf

csv_file_name = './data/period_trend.csv'
reader = tf.contrib.timeseries.CSVReader(csv_file_name)

with tf.Session() as sess:
    data = reader.read_full()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(data))
    coord.request_stop()

train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=4, window_size=16)
with tf.Session() as sess:
    data = train_input_fn.create_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    batch1 = sess.run(data[0])
    batch2 = sess.run(data[0])
    coord.request_stop()

print('batch1:', batch1)
print('batch2:', batch2)
