# coding:utf-8
# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Converts image data to TFRecords file format with Example protos.
The image data set is expected to reside in JPEG files located in the
following directory structure.
  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...
where the sub-directory is the unique label associated with these images.
This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files
  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024
and
  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128
where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'
  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer. start from "class_label_base"
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'
If you data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading


import numpy as np
import tensorflow as tf
import logging


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(text),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with open(filename, 'r') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        logging.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, command_args):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s_%s_%.5d-of-%.5d.tfrecord' % (command_args.dataset_name, name, shard, num_shards)
        output_file = os.path.join(command_args.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                logging.info('%s [thread %d]: Processed %d of %d images in thread batch.' %
                             (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        logging.info('%s [thread %d]: Wrote %d images to %s' %
                     (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    logging.info('%s [thread %d]: Wrote %d images to %d shards.' %
                 (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards, command_args):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), command_args.num_threads + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    logging.info('Launching %d threads for spacings: %s' % (command_args.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards, command_args)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    logging.info('%s: Finished writing all %d images in data set.' %
                 (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file, command_args):
    """Build a list of all images files and labels in the data set.
    Args:
      data_dir: string, path to the root directory of images.
        Assumes that the image data set resides in JPEG files located in
        the following directory structure.
          data_dir/dog/another-image.JPEG
          data_dir/dog/my-image.jpg
        where 'dog' is the label associated with these images.
      labels_file: string, path to the labels file.
        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          dog
          cat
          flower
        where each line corresponds to a label. We map each label contained in
        the file to an integer starting with the integer 0 corresponding to the
        label contained in the first line.
    Returns:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
    """
    logging.info('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    """非常重要，这里我们调整label从0开始以符合定义"""
    label_index = command_args.class_label_base

    # Construct the list of JPEG files and labels.
    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            logging.info('Finished finding files in %d of %d classes.' % (
                label_index, len(labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    logging.info('Found %d JPEG files across %d labels inside %s.' %
                 (len(filenames), len(unique_labels), data_dir))
    # print(labels)
    return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file, command_args):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    filenames, texts, labels = _find_image_files(directory, labels_file, command_args)
    _process_image_files(name, filenames, texts, labels, num_shards, command_args)


def check_and_set_default_args(command_args):
    if not(hasattr(command_args, 'train_shards')) or command_args.train_shards is None:
        command_args.train_shards = 5
    if not(hasattr(command_args, 'validation_shards')) or command_args.validation_shards is None:
        command_args.validation_shards = 5
    if not(hasattr(command_args, 'num_threads')) or command_args.num_threads is None:
        command_args.num_threads = 5
    if not(hasattr(command_args, 'class_label_base')) or command_args.class_label_base is None:
        command_args.class_label_base = 0
    if not(hasattr(command_args, 'dataset_name')) or command_args.dataset_name is None:
        command_args.dataset_name = ''
    assert not command_args.train_shards % command_args.num_threads, (
        'Please make the command_args.num_threads commensurate with command_args.train_shards')
    assert not command_args.validation_shards % command_args.num_threads, (
        'Please make the command_args.num_threads commensurate with '
        'command_args.validation_shards')
    assert command_args.train_directory is not None
    assert command_args.validation_directory is not None
    assert command_args.labels_file is not None
    assert command_args.output_directory is not None


def main(command_args):
    """
    command_args:需要有以下属性：
    command_args.train_directory  训练集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
    command_args.validation_directory 验证集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
    command_args.labels_file 一个文件。每一行代表一个label名称。
    command_args.output_directory 一个文件夹，表示最后输出的位置。

    command_args.train_shards 将训练集分成多少份。
    command_args.validation_shards 将验证集分成多少份。
    command_args.num_threads 线程数。必须是上面两个参数的约数。

    command_args.class_label_base 很重要！真正的tfrecord中，每个class的label号从多少开始，默认为0（在models/slim中就是从0开始的）
    command_args.dataset_name 字符串，输出的时候的前缀。

    图片不可以有损坏。否则会导致线程提前退出。
    """
    check_and_set_default_args(command_args)
    logging.info('Saving results to %s' % command_args.output_directory)

    # Run it!
    _process_dataset('validation', command_args.validation_directory,
                     command_args.validation_shards, command_args.labels_file, command_args)
    _process_dataset('train', command_args.train_directory,
                     command_args.train_shards, command_args.labels_file, command_args)
