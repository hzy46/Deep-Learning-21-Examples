import tensorflow as tf
from glob import glob
import os
import argparse
import logging
from PIL import Image
import traceback

def glob_all(dir_path):
    pic_list = glob(os.path.join(dir_path, '*.jpg'))
    inside = os.listdir(dir_path)
    for dir_name in inside:
        if os.path.isdir(os.path.join(dir_path, dir_name)):
            pic_list.extend(glob_all(os.path.join(dir_path, dir_name)))
    return pic_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dir-path', default='data/')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    all_pic_list = glob_all(args.dir_path)
    for i, img_path in enumerate(all_pic_list):
        try:
            sess = tf.Session()
            with open(img_path, 'rb') as f:
                img_byte = f.read()
                img = tf.image.decode_jpeg(img_byte)
                data = sess.run(img)
                if data.shape[2] != 3:
                    print(data.shape)
                    raise Exception
            tf.reset_default_graph()
            img = Image.open(img_path)
        except Exception:
            logging.warning('%s has broken. Delete it.' % img_path)
            # logging.error(traceback.format_exc())
            os.remove(img_path)
        if (i + 1) % 1000 == 0:
            logging.info('Processing %d / %d.' % (i + 1, len(all_pic_list)))
