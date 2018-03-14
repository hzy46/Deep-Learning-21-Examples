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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

class NodeLookup(object):
  def __init__(self, label_lookup_path=None):
    self.node_lookup = self.load(label_lookup_path)

  def load(self, label_lookup_path):
    node_id_to_name = {}
    with open(label_lookup_path) as f:
      for index, line in enumerate(f):
        node_id_to_name[index] = line.strip()
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  with tf.Graph().as_default():
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    image_data = tf.image.decode_jpeg(image_data)
    image_data = preprocess_for_eval(image_data, 299, 299)
    image_data = tf.expand_dims(image_data, 0)
    with tf.Session() as sess:
      image_data = sess.run(image_data)

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
    predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(FLAGS.label_path)

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))


def main(_):
  image = FLAGS.image_file
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_path',
      type=str,
  )
  parser.add_argument(
      '--label_path',
      type=str,
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
