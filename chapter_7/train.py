# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time
import losses
import utils
import os
import argparse

slim = tf.contrib.slim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/mosaic.yml', help='the path to the conf file')
    return parser.parse_args()


def main(FLAGS):
    style_features_t = losses.get_style_features(FLAGS)

    # Make sure the training path exists.
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """Build Network"""
            network_fn = nets_factory.get_network_fn(
                FLAGS.loss_model,
                num_classes=1,
                is_training=False)

            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                                            'train2014/', image_preprocessing_fn, epochs=FLAGS.epoch)
            generated = model.net(processed_images, training=True)
            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            processed_generated = tf.stack(processed_generated)
            _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)

            # Log the structure of loss network
            tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
            for key in endpoints_dict:
                tf.logging.info(key)

            """Build Losses"""
            content_loss = losses.content_loss(endpoints_dict, FLAGS.content_layers)
            style_loss, style_loss_summary = losses.style_loss(endpoints_dict, style_features_t, FLAGS.style_layers)
            tv_loss = losses.total_variation_loss(generated)  # use the unprocessed image

            loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss

            # Add Summary for visualization in tensorboard.
            """Add Summary"""
            tf.summary.scalar('losses/content_loss', content_loss)
            tf.summary.scalar('losses/style_loss', style_loss)
            tf.summary.scalar('losses/regularizer_loss', tv_loss)

            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * FLAGS.content_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss * FLAGS.style_weight)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * FLAGS.tv_weight)
            tf.summary.scalar('total_loss', loss)

            for layer in FLAGS.style_layers:
                tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            tf.summary.image('generated', generated)
            # tf.image_summary('processed_generated', processed_generated)  # May be better?
            tf.summary.image('origin', tf.stack([
                image_unprocessing_fn(image) for image in tf.unstack(processed_images, axis=0, num=FLAGS.batch_size)
            ]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)

            """Prepare to Train"""
            global_step = tf.Variable(0, name="global_step", trainable=False)

            variable_to_train = []
            for variable in tf.trainable_variables():
                if not(variable.name.startswith(FLAGS.loss_model)):
                    variable_to_train.append(variable)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            variables_to_restore = []
            for v in tf.global_variables():
                if not(v.name.startswith(FLAGS.loss_model)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Restore variables for loss network.
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            # Restore variables for training model if the checkpoint file exists.
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            """Start Training"""
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    # print(step)
                    if step % 10 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)
