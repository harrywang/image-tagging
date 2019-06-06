# !/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time

import os.path
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

flags = tf.app.flags
FLAGS = flags.FLAGS

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/pretrain_open_images'

flags.DEFINE_string('labelmap', BASE_DIR + '/classes-trainable.txt',
                    'Labels, one per line.')

# Total 5000 labels (classes/categories)
flags.DEFINE_string('dict', BASE_DIR + '/class-descriptions.csv',
                    'Descriptive string for each label.')

flags.DEFINE_string('checkpoint_path', BASE_DIR + '/oidv2-resnet_v1_101.ckpt',
                    'Path to checkpoint file.')

# After tagging, each image will get a list like this:
# [(label_1, probability_of_label_1),...,(label_5000, probability_of_label_5000)]

# Config to return top-k labels with highest probabilities
flags.DEFINE_integer('top_k', 10, 'Maximum number of results to show.')

# Only return labels with probability > score_threshold
flags.DEFINE_float('score_threshold', 0.1, 'Score threshold.')


class AutoTagEngine:

    def __init__(self):
        return

    label_map = None
    label_dict = None
    label_5000_list = []

    flags = FLAGS

    @staticmethod
    def init_data():
        if AutoTagEngine.label_map is None:
            AutoTagEngine.label_map = [line.rstrip() for line in tf.gfile.GFile(AutoTagEngine.flags.labelmap)]
        if AutoTagEngine.label_dict is None:
            AutoTagEngine.label_dict = {}
            for line in tf.gfile.GFile(AutoTagEngine.flags.dict):
                words = [word.strip(' "\n') for word in line.split(',', 1)]
                AutoTagEngine.label_dict[words[0]] = words[1]

            for label_id in AutoTagEngine.label_map:
                AutoTagEngine.label_5000_list.append(AutoTagEngine.label_dict[label_id])

    @staticmethod
    def make_batch_predictions(images_list_urls, batch_size=1000):

        total_images = len(images_list_urls)

        num_batch = math.floor(total_images / batch_size)
        if total_images % batch_size > 0:
            num_batch += 1

        # result = {image_name:[prob1, prob2,...,prob5000], ...]}
        result = {}

        for index in range(int(num_batch)):

            start = index * batch_size
            stop = start + batch_size

            if stop > total_images:
                stop = total_images

            # Get batch of images
            image_list_files = images_list_urls[start:stop]

            start_tagging_time = time.time()
            batch_predictions_result = AutoTagEngine.make_predictions(image_list_files)
            elapsed_time = time.time() - start_tagging_time

            tagged_image_count = batch_size
            if batch_size > total_images:
                tagged_image_count = total_images

            print('Finish tagging {0} images in batch {1}/{2} with {3} seconds'.format(tagged_image_count, (index + 1),
                                                                                       int(num_batch), elapsed_time))
            result.update(batch_predictions_result)

        return result

    @staticmethod
    def make_predictions(image_list_files):

        predictions_dict = {}

        g = tf.Graph()
        with g.as_default():
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(AutoTagEngine.flags.checkpoint_path + '.meta')
                saver.restore(sess, AutoTagEngine.flags.checkpoint_path)

                input_values = g.get_tensor_by_name('input_values:0')
                predictions = g.get_tensor_by_name('multi_predictions:0')

                for image_filename in image_list_files:

                    if os.path.isfile(image_filename) is False:
                        continue

                    try:
                        compressed_image = tf.gfile.FastGFile(image_filename, 'rb').read()
                        predictions_eval = sess.run(
                            predictions, feed_dict={
                                input_values: [compressed_image]
                            })

                        predictions_dict[image_filename] = predictions_eval

                    except KeyboardInterrupt:
                        raise
                    except:
                        print('Error with image: "%s"\n' % image_filename)

        return predictions_dict

    @staticmethod
    def do_tagging_process(images):
        AutoTagEngine.init_data()
        predict_result = AutoTagEngine.make_batch_predictions(images)
        return predict_result
