import sys
import csv
import glob
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm

FEATURES_NUMBER = 1000
GPU = int(sys.argv[1])

input_features = tf.placeholder(
    tf.float32, shape=[None, FEATURES_NUMBER], name="input_features")
features_batch = tf.placeholder(
    tf.float32, shape=[None, FEATURES_NUMBER], name="features_batch")

similarity = tf.reduce_sum(
    tf.square(
        tf.nn.l2_normalize(input_features, 1) -
        tf.nn.l2_normalize(features_batch, 1)),
    reduction_indices=1,
    keep_dims=False)

labels_filenames = sorted(glob.glob("db/train-labels*.npy"))
features_filenames = sorted(glob.glob("db/train-features*.npy"))

j = 0
with tf.Session() as sess:
    for labels_filename, features_filename in zip(labels_filenames,
                                                  features_filenames):
        j += 1
        ofh = open('knn/results_%s_%s.txt' % (GPU, j), "w")
        writer = csv.writer(ofh)

        data_labels = np.load(labels_filename)
        data_features = np.load(features_filename)[:, 0:FEATURES_NUMBER]

        print features_filename
        tfh = open("bin/outdoor-test-features.csv.%s" % GPU, "r")
        for line in tqdm(tfh):

            input_csv = np.array(
                line.split(',')[1:(FEATURES_NUMBER + 1)]).astype('float32')
            input_label = line.split(',')[0]

            results = []

            tile_input_csv = np.tile(input_csv, (len(data_features), 1))
            similarity_ = sess.run(
                similarity,
                feed_dict={
                    input_features: tile_input_csv,
                    features_batch: data_features
                })
            for i in range(len(data_features)):
                row = [str(data_labels[i])] + [str(similarity_[i])]
                results.append(row)

            results = sorted(results, key=lambda results: float(results[1]))[0]
            writer.writerow([input_label] + results)
        ofh.close()
