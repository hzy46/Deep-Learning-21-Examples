# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os

import model
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', help='the path to the model file')
    parser.add_argument('-n', '--model_name', default='transfer', help='the name of the model')
    parser.add_argument('-d', dest='is_debug', action='store_true')
    parser.set_defaults(is_debug=False)
    return parser.parse_args()


def main(args):
    g = tf.Graph()      # A new graph
    with g.as_default():
        with tf.Session() as sess:
            # Building graph.
            image_data = tf.placeholder(tf.int32, name='input_image')
            height = tf.placeholder(tf.int32, name='height')
            width = tf.placeholder(tf.int32, name='width')

            # Reshape data
            image = tf.reshape(image_data, [height, width, 3])

            processed_image = utils.mean_image_subtraction(
                image, [123.68, 116.779, 103.939])                    # Preprocessing image
            batched_image = tf.expand_dims(processed_image, 0)        # Add batch dimension
            generated_image = model.net(batched_image, training=False)
            casted_image = tf.cast(generated_image, tf.int32)
            # Remove batch dimension
            squeezed_image = tf.squeeze(casted_image, [0])
            cropped_image = tf.slice(squeezed_image, [0, 0, 0], [height, width, 3])
            # stylized_image = tf.image.encode_jpeg(squeezed_image, name='output_image')
            stylized_image_data = tf.reshape(cropped_image, [-1], name='output_image')

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path.
            model_file = os.path.abspath(args.model_file)
            saver.restore(sess, model_file)

            if args.is_debug:
                content_file = '/Users/Lex/Desktop/t.jpg'
                generated_file = '/Users/Lex/Desktop/xwz-stylized.jpg'

                with open(generated_file, 'wb') as img:
                    image_bytes = tf.read_file(content_file)
                    input_array, decoded_image = sess.run([
                        tf.reshape(tf.image.decode_jpeg(image_bytes, channels=3), [-1]),
                        tf.image.decode_jpeg(image_bytes, channels=3)])

                    start_time = time.time()
                    img.write(sess.run(tf.image.encode_jpeg(tf.cast(cropped_image, tf.uint8)), feed_dict={
                              image_data: input_array,
                              height: decoded_image.shape[0],
                              width: decoded_image.shape[1]}))
                    end_time = time.time()

                    tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
            else:
                output_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess, sess.graph_def, output_node_names=['output_image'])

                with tf.gfile.FastGFile('/Users/Lex/Desktop/' + args.model_name + '.pb', mode='wb') as f:
                    f.write(output_graph_def.SerializeToString())

                # tf.train.write_graph(g.as_graph_def(), '/Users/Lex/Desktop',
                #                      args.model_name + '.pb', as_text=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    print(args)
    main(args)
