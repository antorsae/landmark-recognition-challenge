# (C) 2018 Andres Torrubia, licensed under GNU General Public License v3.0 
# See license.txt

import argparse
import glob
import numpy as np
import pandas as pd
import random
from os.path import join
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, AveragePooling2D, Reshape, SeparableConv2D
from keras.utils import to_categorical
from keras.applications import *
from keras import backend as K
from keras.engine.topology import Layer
import keras.losses
from keras.utils import CustomObjectScope

from multi_gpu_keras import multi_gpu_model

import skimage
from iterm import show_image

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import copy
import itertools
import re
import os
import sys
import jpeg4py as jpeg
from scipy import signal
import cv2
import math
import csv
from multiprocessing import Pool
from multiprocessing import cpu_count, Process, Queue, JoinableQueue, Lock

from functools import partial
from itertools import  islice
from conditional import conditional

from collections import defaultdict
import copy

import imgaug as ia
from imgaug import augmenters as iaa
import sharedmem
from hadamard import HadamardClassifier
from clr_callback import CyclicLR
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
# TODO tf seed

parser = argparse.ArgumentParser()
# general
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-g', '--gpus', type=int, default=None, help='Number of GPUs to use')
parser.add_argument('-v', '--verbose', action='store_true', help='Pring debug/verbose info')
parser.add_argument('-b', '--batch-size', type=int, default=48, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning-rate', type=float, default=None, help='Initial learning rate')
parser.add_argument('-clr', '--cyclic_learning_rate',action='store_true', help='Use cyclic learning rate https://arxiv.org/abs/1506.01186')
parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer to use in training -o adam|sgd|adadelta')
parser.add_argument('--amsgrad', action='store_true', help='Apply the AMSGrad variant of adam|adadelta from the paper "On the Convergence of Adam and Beyond".')

# architecture/model
parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
parser.add_argument('-do', '--dropout', type=float, default=0., help='Dropout rate for first FC layer')
parser.add_argument('-dol', '--dropout-last', type=float, default=0., help='Dropout rate for last FC layer')
parser.add_argument('-doc', '--dropout-classifier', type=float, default=0., help='Dropout rate for classifier')
parser.add_argument('-nfc', '--no-fcs', action='store_true', help='Dont add any FC at the end, just a softmax')
parser.add_argument('-fc', '--fully-connected-layers', nargs='+', type=int, default=[512,256], help='Specify FC layers after classifier, e.g. -fc 1024 512 256')
parser.add_argument('-f', '--freeze', type=int, default=0, help='Freeze first n CNN layers, e.g. --freeze 10')
parser.add_argument('-fu', '--freeze-until', type=str, default=None, help='Freeze until named CNN layer, e.g. --freeze-until 10')
parser.add_argument('-fca', '--fully-connected-activation', type=str, default='relu', help='Activation function to use in FC layers, e.g. -fca relu|selu|prelu|leakyrelu|elu|...')
parser.add_argument('-bn', '--batch-normalization', action='store_true', help='Use batch normalization in FC layers')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-pcs', '--print-classifier-summary', action='store_true', help='Print classifier model summary')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
parser.add_argument('-rp', '--reduce-pooling', type=int, default=None, help='If using pooling none add conv layers to reduce features, e.g. -rp 128')
parser.add_argument('-lo', '--loss', type=str, default='categorical_crossentropy', help='Loss function')
parser.add_argument('-hp', '--hadamard', action='store_true', help='Use Hadamard projection instead of FC layers, see https://arxiv.org/pdf/1801.04540.pdf')
parser.add_argument('-l2', '--l2-normalize', action='store_true', help='Perform L2 normalization in Hadamard classifier')
parser.add_argument('-pp', '--post-pooling', type=str, default=None, help='Add pooling layers after classifier, e.g. -pp avg|max')
parser.add_argument('-pps', '--post-pool-size', type=int, default=2, help='Pooling factor for pooling layers after classifier, e.g. -pps 3')
parser.add_argument('-dl', '--delete-layers', nargs='+', type=str, default=[], help='Specify layers to delete in classifier, e.g. -dl avg_pool')

# training regime
parser.add_argument('-cs', '--crop-size', type=int, default=256, help='Crop size')
parser.add_argument('-vpc', '--val-percent', type=float, default=0.15, help='Val percent')
parser.add_argument('-cc', '--center-crops', nargs='*', type=int, default=[], help='Train on center crops only (not random crops) for the selected classes e.g. -cc 1 6 or all -cc -1')
parser.add_argument('-ap', '--augmentation-probability', type=float, default=1., help='Probability of augmentation after 1st seen sample')
parser.add_argument('-fcm', '--freeze-classifier', action='store_true', help='Freeze classifier weights (useful to fine-tune FC layers)')
# training regime (class aware sampling options)
parser.add_argument('-cas', '--class-aware-sampling', action='store_true', help='Use class aware sampling to balance dataset (instead of class weights)')
parser.add_argument('-casac', '--class-aware-sampling-accuracy-target', type=float, default=0.9, help='Threshold to move to next landmark group (when using -cas)')
parser.add_argument('-casab', '--class-aware-sampling-accuracy-batches', type=int, default=20, help='Number of batches to check for threshold to move to next landmark group (when using -cas)')
parser.add_argument('-casp',  '--class-aware-sampling-patience', type=int, default=8000, help='Patience in number of items to move to nexty landmark_to_cat (when using -cas)')
parser.add_argument('-casr', '--class-aware-sampling-resume', type=int, default=0, help='Resume from group, e.g. -casr 8 (group starts at 0)')

# dataset (training)
parser.add_argument('-id', '--include-distractors', action='store_true', help='Include distractors from retrieval challenge') # DO NOT USE YET

# test
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV/npy submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-th', '--threshold', default=0., type=float, help='Ignore predictions less than threshold, e.g. -th 0.6')

args = parser.parse_args()

training = not (args.test or args.test_train)

if not args.verbose:
    import warnings
    warnings.filterwarnings("ignore")

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if args.gpus is None:
    args.gpus = len(get_available_gpus())   

args.batch_size *= args.gpus

TRAIN_DIR    = 'train-dl'
TRAIN_JPGS   = set(Path(TRAIN_DIR).glob('*.jpg'))
TRAIN_IDS    = { os.path.splitext(os.path.basename(item))[0] for item in TRAIN_JPGS }

if args.test:
    TEST_DIR     = 'test-dl'
    TEST_JPGS    = list(Path(TEST_DIR).glob('*.jpg'))
    TEST_IDS     = { os.path.splitext(os.path.basename(item))[0] for item in TEST_JPGS  }

MODEL_FOLDER        = 'models'
CSV_FOLDER          = 'csv'
TRAIN_CSV           = 'train.csv'
TEST_CSV            = 'test.csv'

if args.include_distractors:
    DISTRACTOR_JPGS   = list(Path('distractors').glob('*.jpg'))
    DISTRACTOR_IDS    = { os.path.splitext(os.path.basename(item))[0] for item in DISTRACTOR_JPGS }

CROP_SIZE = args.crop_size

id_to_landmark  = { }
id_to_cat       = { }
id_times_seen   = { }

landmark_to_ids = defaultdict(list)
cat_to_ids      = defaultdict(list)
landmark_to_cat = { }
cat_to_landmark = { }
# since we may get holes in landmark (ids) from the CSV file
# we'll use cat (category) starting from 0 and keep a few dicts to map around
cat = 0
with open(TRAIN_CSV, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        idx, landmark = row[0][1:-1], int(row[2])
        if idx in TRAIN_IDS:
            if landmark in landmark_to_cat:
                landmark_cat = landmark_to_cat[landmark]
            else:
                landmark_cat = cat
                landmark_to_cat[landmark] = landmark_cat
                cat_to_landmark[cat] = landmark
                cat += 1 
            id_to_landmark[idx] = landmark
            id_to_cat[idx]      = landmark_cat
            id_times_seen[idx]  = 0
            landmark_to_ids[landmark].append(idx)
            cat_to_ids[landmark_cat].append(idx)

if args.include_distractors:
    landmark = -1
    landmark_cat = cat
    landmark_to_cat[landmark] = landmark_cat
    cat_to_landmark[landmark_cat] = landmark

    for idx in DISTRACTOR_IDS:
        id_to_landmark[idx] = landmark
        id_to_cat[idx]      = landmark_cat
        id_times_seen[idx]  = 0
        landmark_to_ids[landmark].append(idx)
        cat_to_ids[landmark_cat].append(idx)

N_CLASSES = len(landmark_to_cat.keys())

print(len(id_to_landmark.keys()), N_CLASSES)

def get_class(item):
    return id_to_cat[os.path.splitext(os.path.basename(item))[0]]

def get_id(item):
    return os.path.splitext(os.path.basename(item))[0]

# since we are doing stratified train/val split we need to dupe images
# from landmarks wirth just 1 item
ids_to_dup = [ids[0] for cat,ids in cat_to_ids.items() if len(ids) == 1]

print(len(ids_to_dup))

TRAIN_JPGS = list(TRAIN_JPGS) + ids_to_dup 

if args.include_distractors:
    TRAIN_JPGS += DISTRACTOR_JPGS

    print("Total items in set {} of which {:.2f}% are distractors".format(
        len(TRAIN_JPGS), 
        100. * len(DISTRACTOR_JPGS) / len(TRAIN_JPGS)))
else:
    print("Total items in set {}".format(
        len(TRAIN_JPGS), ))

TRAIN_CATS = [ get_class(idx) for idx in TRAIN_JPGS ]

def preprocess_image(img):
    
    # find `preprocess_input` function specific to the classifier
    classifier_to_module = { 
        'NASNetLarge'       : 'nasnet',
        'NASNetMobile'      : 'nasnet',
        'DenseNet121'       : 'densenet',
        'DenseNet161'       : 'densenet',
        'DenseNet201'       : 'densenet',
        'InceptionResNetV2' : 'inception_resnet_v2',
        'InceptionV3'       : 'inception_v3',
        'MobileNet'         : 'mobilenet',
        'ResNet50'          : 'resnet50',
        'VGG16'             : 'vgg16',
        'VGG19'             : 'vgg19',
        'Xception'          : 'xception',

        'SEDenseNetImageNet121' : 'se_densenet',
        'SEDenseNetImageNet161' : 'se_densenet',
        'SEDenseNetImageNet169' : 'se_densenet',
        'SEDenseNetImageNet264' : 'se_densenet',
        'SEInceptionResNetV2'   : 'se_inception_resnet_v2',
        'SEMobileNet'           : 'se_mobilenets',
        'SEResNet50'            : 'se_resnet',
        'SEResNet101'           : 'se_resnet',
        'SEResNet154'           : 'se_resnet',
        'SEInceptionV3'         : 'se_inception_v3',
        'SEResNext'             : 'se_resnet',
        'SEResNextImageNet'     : 'se_resnet',

    }

    if args.classifier in classifier_to_module:
        classifier_module_name = classifier_to_module[args.classifier]
    else:
        classifier_module_name = 'xception'

    preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
    return preprocess_input_function(img.astype(np.float32))

def augment(img):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            # crop images by -5% to 10% of their height/width
            iaa.Crop(
                percent=(0, 0.2),
            ),
            iaa.Scale({"height": CROP_SIZE, "width": CROP_SIZE }),
        ],
        random_order=False
    )

    if img.ndim == 3:
        img = seq.augment_images(np.expand_dims(img, axis=0)).squeeze(axis=0)
    else:
        img = seq.augment_images(img)

    return img

# reads the image referenced by item from disk and
# returns img, one_hot_class_idx, item
# img: processed image (normalized as excepted by NN) and augmented if both aug and training are True
# one_hot_class_idx: one-hot vector of cat id (if predict is false)
# item: same as passed 
# 
# img and one_hot_class_idx will be None if error reading item
#
def process_item(item, aug = False, training = False, predict=False):

    load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
    load_img           = lambda img_path: np.array(Image.open(img_path))

    def try_load_PIL(item):
        try:
            img = load_img(item)
            return img
        except Exception:
            if args.verbose:
                print('Decoding error:', item)
            return None

    validation = not training 

    loaded_pil = loaded_fast_jpg = False
    try:
        img = load_img_fast_jpg(item)
        loaded_fast_jpg = True
    except Exception:
        img = try_load_PIL(item)
        if img is None: return None, None, item
        loaded_pil = True

    shape = list(img.shape[:2])

    # some images may not be downloaded correctly and are B/W, discard those
    if img.ndim != 3:
        if args.verbose:
            print('Ndims !=3 error:', item)
        if not loaded_pil:
            img = try_load_PIL(item)
            if img is None: return None, None, item
            loaded_pil = True
        if img.ndim == 2:
            img = np.stack((img,)*3, -1)
        if img.ndim != 3:
            return None, None, item

    if img.shape[2] != 3:
        if args.verbose:
            print('More than 3 channels error:', item)
        if not loaded_pil:
            img = try_load_PIL(item)
            if img is None: return None, None, item
            loaded_pil = True   
        return None, None, item

    if training and aug and np.random.random() < args.augmentation_probability:
        img = augment(img)
        if np.random.random() < 0.0:
            show_image(img)
    else:
        img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))

    img = preprocess_image(img)

    if args.verbose:
        print("ap: ", img.shape, item)

    if not predict:
        one_hot_class_idx = to_categorical(get_class(item), N_CLASSES)
    else:
        one_hot_class_idx = np.zeros(N_CLASSES, dtype=np.float32)

    return img, one_hot_class_idx, item

# multiprocess worker to read items and put them in shared memory for consumer
def process_item_worker(worker_id, lock, shared_mem_X, shared_mem_y, jobs, results):
    # make sure augmentations are different for each worker
    np.random.seed()
    random.seed()

    while True:
        item, aug, training, predict = jobs.get()
        img, one_hot_class_idx, item = process_item(item, aug, training, predict)
        is_good_item = False
        if one_hot_class_idx is not None:
            lock.acquire()
            shared_mem_X[worker_id,...] = img
            shared_mem_y[worker_id,...] = one_hot_class_idx
            is_good_item = True
        results.put((worker_id, is_good_item, item))

# Callback to monitor accuracy on a per-batch basis
class AccuracyReset(Callback):

    N_BATCHES = args.class_aware_sampling_accuracy_batches

    def __init__(self, filepath):
        super(AccuracyReset, self).__init__()
        self.filepath = filepath

    def on_train_begin(self, logs={}):

        self.aucs = []
        self.losses = []
        self.reset_accuracy()
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        return
 
    def on_batch_begin(self, batch, logs={}):
        #print(logs)
        return
 
    def on_batch_end(self, batch, logs={}):
        self.last_accuracies[self.last_accuracies_i % AccuracyReset.N_BATCHES ] = logs['categorical_accuracy']
        self.last_accuracies_i += 1
        #print( self.last_accuracies)
        if np.all(self.last_accuracies >= args.class_aware_sampling_accuracy_target):
            self.accuracy_reached = True
        return

    def reset_accuracy(self, group=-1, save = False):
        self.accuracy_reached = False
        self.last_accuracies = np.zeros(AccuracyReset.N_BATCHES)
        self.last_accuracies_i = 0
        if group != -1 and save:
            self.model.save(
                self.filepath.format(group= group, epoch= self.epoch + 1), 
                overwrite=True)
        return

# main generator. Although predict=True mode works it is not used here.
def gen(items, batch_size, training=True, predict=False, accuracy_callback=None):

    validation = not training 
    items_set = set(items)

    # X image crops
    X = np.empty((batch_size, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

    if predict:
        training = False

    # class index
    y = np.empty((batch_size, N_CLASSES),               dtype=np.float32)
    
    n_group_classes = int(math.ceil(N_CLASSES / batch_size))
    if training and args.class_aware_sampling:
        items_per_class = defaultdict(list)
        n_items_per_class = np.zeros(N_CLASSES, dtype=np.int64)
        for item in items:
            class_idx = get_class(item)
            items_per_class[class_idx].append(item)
            n_items_per_class[class_idx] += 1
        items_per_class_running=copy.deepcopy(items_per_class)
        classes_groups = np.array_split(np.argsort(n_items_per_class)[::-1], n_group_classes)
        classes_current_group = -1
        classes_current_group_items_to_see = 0
        classes_running_copy  = [ ]
        classes_seen = set()
        previous_classes_seen = set()
        if args.class_aware_sampling_resume != 0:
            # if resuming to group args.class_aware_sampling_resume:
            for class_group in classes_groups[:args.class_aware_sampling_resume]:
                # add classes from previous groups to previously seen classes
                previous_classes_seen = previous_classes_seen.union(class_group)
                # and for each class in group
                for class_idx in class_group:
                    # mark items up to worst case scenario (patience reached) as seen once so augmentation kicks in
                    for item in items_per_class[class_idx][:args.class_aware_sampling_patience]:
                        id_times_seen[get_id(item)] += 1
            classes_current_group = (args.class_aware_sampling_resume - 1) % n_group_classes
            print("Resuming from group {}. Landmarks marked as seen: {}".format(
                classes_current_group,
                " ".join([str(cat_to_landmark[cat]) for cat in previous_classes_seen])))
        # dont save model the first time
        save_model = False

    n_workers    = (cpu_count() - 1) if not predict else 1 # for prediction we need to guarantee order
    shared_mem_X = sharedmem.empty((n_workers, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
    shared_mem_y = sharedmem.empty((n_workers, N_CLASSES),               dtype=np.float32)
    locks        = [Lock()] * n_workers
    jobs         = Queue(args.batch_size * 4 if not predict else 1)
    results      = JoinableQueue(args.batch_size * 2 if not predict else 1)

    [Process(
        target=process_item_worker, 
        args=(worker_id, lock, shared_mem_X, shared_mem_y, jobs, results)).start() for worker_id, lock in enumerate(locks)]

    bad_items = set()
    i = 0

    while True:

        if training and not args.class_aware_sampling:
            random.shuffle(items)

        batch_idx = 0

        items_done  = 0
        while items_done < len(items):  
            # fill the queue to make sure CPU is always busy
            while not jobs.full():
                if training and args.class_aware_sampling:
                    if np.random.rand() >= 0.2 or not previous_classes_seen:
                        # if already reached patience or accuracy, build classes list with this group's classes
                        if classes_current_group_items_to_see == 0 or accuracy_callback.accuracy_reached:
                            print(accuracy_callback.last_accuracies)
                            accuracy_callback.reset_accuracy(classes_current_group, save = save_model)
                            # save model from now on
                            save_model = True
                            classes_current_group_items_to_see = int(args.class_aware_sampling_patience * N_CLASSES / n_group_classes)
                            classes_current_group = (classes_current_group + 1) % n_group_classes
                            classes = list(classes_groups[classes_current_group])
                            classes_running_copy = []
                            previous_classes_seen = previous_classes_seen.union(classes_seen)
                            classes_seen = set()
                            print("Class group #{} {}/{} ({:.2f}% of items)".format(
                                classes_current_group, 
                                classes_current_group+1, 
                                n_group_classes,
                                100. * sum([n_items_per_class[class_id] for class_id in previous_classes_seen.union(set(classes))]) / len(items),
                                ))
                        # if we've run out of items in this class, replenish it
                        if len(classes_running_copy) == 0:
                            random.shuffle(classes)
                            classes_running_copy = list(classes)
                        random_class = classes_running_copy.pop()
                        classes_current_group_items_to_see -= 1
                    else:
                        # pick a random class of the ones seen previously to make sure the net
                        # doesn't forget what it's learned
                        random_class = random.sample(previous_classes_seen, 1)[0]

                    # pick an item from class making sure it belongs to the items
                    # this is not strictly needed, though
                    training_item_chosen = False
                    while not training_item_chosen:
                        if len(items_per_class_running[random_class]) == 0:
                            random.shuffle(items_per_class_running[random_class])
                            items_per_class_running[random_class]=copy.deepcopy(items_per_class[random_class])
                        item = items_per_class_running[random_class].pop()
                        training_item_chosen = item in items_set
                    classes_seen.add(random_class)
                else:
                    # if not using class-aware sampling, just pick one item
                    item = items[i % len(items)]
                    i += 1
                if not predict:
                    # do not augment the first time the net has seen an item
                    aug = False if id_times_seen[get_id(item)] == 0 else True
                    id_times_seen[get_id(item)] += 1
                else:
                    # do not augment if predicting
                    aug = False
                jobs.put((item, aug, training, predict))
                items_done += 1

            # loop over results and yield until no more resuls left
            get_more_results = True
            while get_more_results:
                worker_id, is_good_item, _item = results.get() # blocks/waits if None
                results.task_done()

                if is_good_item:
                    X[batch_idx], y[batch_idx] = shared_mem_X[worker_id], shared_mem_y[worker_id]
                    locks[worker_id].release()
                    batch_idx += 1
                else:
                    if predict:
                        X[batch_idx] = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
                        batch_idx += 1
                        print("Warning {}".format(_item))
                    bad_items.add(_item)

                if batch_idx == batch_size:
                    if not predict:
                        yield(X, y)
                    else:
                        yield(X)
                    batch_idx = 0

                get_more_results = not results.empty()

        if len(bad_items) > 0:
            print("\nRejected {} items: {}".format('trainining' if training else 'validation', len(bad_items)))

# MAIN
if args.model:
    print("Loading model " + args.model)

    with CustomObjectScope({'HadamardClassifier': HadamardClassifier}):
        model = load_model(args.model, compile=False if not training or (args.learning_rate is not None) else True)
    # e.g. ResNet50-hp-l2-ppavg2-losscategorical_crossentropy-cs256-nofc-doc0.0-do0.0-dol0.0-poolingnone-cas-epoch008-val_acc0.575105.hdf5
    model_basename = os.path.splitext(os.path.basename(args.model))[0]
    model_parts = model_basename.split('-')
    model_name = '-'.join([part for part in model_parts if part not in ['epoch', 'val_acc']])
    args.classifier = model_parts[0]
    CROP_SIZE = args.crop_size  = model.get_input_shape_at(0)[1]
    print("Overriding classifier: {} and crop size: {}".format(args.classifier, args.crop_size))
    last_epoch = int(list(filter(lambda x: x.startswith('epoch'), model_parts))[0][5:])
    print("Last epoch: {}".format(last_epoch))
    print("Model name: " + model_name)

    if args.learning_rate == None and training:
        dummy_model = model
        args.learning_rate = K.eval(model.optimizer.lr)
        print("Resuming with learning rate: {:.2e}".format(args.learning_rate))

    predictions_name = model.outputs[0].name

elif True:
    if args.learning_rate is None:
        args.learning_rate = 1e-4   # default LR unless told otherwise

    last_epoch = 0

    input_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3),  name = 'image' )

    classifier = globals()[args.classifier]

    classifier_model = classifier(
        include_top=False, 
        weights = 'imagenet' if args.use_imagenet_weights else None,
        input_shape=(CROP_SIZE, CROP_SIZE, 3), 
        pooling=args.pooling if args.pooling != 'none' else None)

    for layer in args.delete_layers:
        classifier_model = delete_layer(classifier_model, classifier_model.get_layer(layer))

    trainable = False
    n_trainable = 0

    for i, layer in enumerate(classifier_model.layers):
        if (i >= args.freeze and args.freeze_until is None) or (layer.name == args.freeze_until):
            trainable = True
        if trainable:
            n_trainable += 1
        layer.trainable = trainable

    print("Base model has " + str(n_trainable) + "/" + str(len(classifier_model.layers)) + " trainable layers")

    if args.print_classifier_summary:
        classifier_model.summary()

    x = input_image

    x = classifier_model(x)

    if args.reduce_pooling and x.shape.ndims == 4:
        # reduce feature channels after classifier using convs

        pool_features = int(x.shape[3])

        for it in range(int(math.log2(pool_features/args.reduce_pooling))):

            pool_features //= 2
            x = Conv2D(pool_features, (3, 3), padding='same', use_bias=False, name='reduce_pooling{}'.format(it))(x)
            x = BatchNormalization(name='bn_reduce_pooling{}'.format(it))(x)
            x = Activation('relu', name='relu_reduce_pooling{}'.format(it))(x)
        
    if x.shape.ndims > 2:
        # reduce spatial channels after classifier using pooling
        if args.post_pooling == 'avg':
            x = AveragePooling2D(pool_size=args.post_pool_size)(x)
        elif args.post_pooling == 'max':
            x = MaxPooling2D(pool_size=args.post_pool_size)(x)

        x = Reshape((-1,), name='reshape0')(x)

    if args.dropout_classifier != 0.:
        x = Dropout(args.dropout_classifier, name='dropout_classifier')(x)

    if not args.no_fcs and not args.hadamard:

        # regular FC classifier
        dropouts = np.linspace( args.dropout,  args.dropout_last, len(args.fully_connected_layers))

        x_m = x

        for i, (fc_layer, dropout) in enumerate(zip(args.fully_connected_layers, dropouts)):
            if args.batch_normalization:
                x_m = Dense(fc_layer//2, name= 'fc_m{}'.format(i))(x_m)
                x_m = BatchNormalization(name= 'bn_m{}'.format(i))(x_m)
                x_m = Activation(args.fully_connected_activation, 
                                         name= 'act_m{}{}'.format(args.fully_connected_activation,i))(x_m)
            else:
                x_m = Dense(fc_layer//2, activation=args.fully_connected_activation, 
                                         name= 'fc_m{}'.format(i))(x_m)
            if dropout != 0:
                x_m = Dropout(dropout,   name= 'dropout_fc_m{}_{:04.2f}'.format(i, dropout))(x_m)

        for i, (fc_layer, dropout) in enumerate(zip(args.fully_connected_layers, dropouts)):
            if args.batch_normalization:
                x = Dense(fc_layer,    name= 'fc{}'.format(i))(x)
                x = BatchNormalization(name= 'bn{}'.format(i))(x)
                x = Activation(args.fully_connected_activation, name='act{}{}'.format(args.fully_connected_activation,i))(x)
            else:
                x = Dense(fc_layer, activation=args.fully_connected_activation, name= 'fc{}'.format(i))(x)
            if dropout != 0:
                x = Dropout(dropout,                   name= 'dropout_fc{}_{:04.2f}'.format(i, dropout))(x)


    if args.hadamard:
        # ignore unscaled logits for now (_)
        x, _ = HadamardClassifier(N_CLASSES, name= "logits", l2_normalize=args.l2_normalize, output_raw_logits=True)(x)
    else:
        x    = Dense(             N_CLASSES, name= "logits")(x)

    activation ="softmax" if args.loss == 'categorical_crossentropy' else "sigmoid"

    print("Using {} activation and {} loss for predictions". format(activation, args.loss))

    prediction = Activation(activation, name="predictions")(x)

    model = Model(inputs=(input_image), outputs=(prediction))

    model_name = args.classifier + \
        ('-hp' if args.hadamard else '') + \
        ('-l2' if args.l2_normalize else '-nol2' if args.hadamard else '') + \
        ('-pp{}{}'.format(args.post_pooling, args.post_pool_size) if args.post_pooling else '') + \
        '-loss{}'.format(args.loss) + \
        '-cs{}'.format(args.crop_size) + \
        ('-fc{}'.format(','.join([str(fc) for fc in args.fully_connected_layers])) if not args.no_fcs else '-nofc') + \
        ('-bn' if args.batch_normalization else '') + \
        (('-doc' + str(args.dropout_classifier)) if args.dropout_classifier != 0. else '') + \
        (('-do'  + str(args.dropout)) if args.dropout != 0. else '') + \
        (('-dol' + str(args.dropout_last)) if args.dropout_last != 0. else '') + \
        ('-pooling' + args.pooling) + \
        ('-id' if args.include_distractors else '') + \
        ('-cc{}'.format(','.join([str(c) for c in args.center_crops])) if args.center_crops else '') + \
        ('-cas' if args.class_aware_sampling else '')

    print("Model name: " + model_name)

    if args.weights:
            model.load_weights(args.weights, by_name=True, skip_mismatch=True)
            match = re.search(r'([,A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.weights)
            last_epoch = int(match.group(2))

model.summary()
model = multi_gpu_model(model, gpus=args.gpus)

if training:

    # split train/val using stratification
    ids_train, ids_val, _, _ = train_test_split(
        TRAIN_JPGS, TRAIN_CATS, test_size=args.val_percent, random_state=SEED, stratify=TRAIN_CATS)

    # compute class weight if not using class-aware sampling
    classes_train = [get_class(idx) for idx in ids_train]
    class_weight = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)

    if args.optimizer == 'adam':
        opt = Adam(lr=args.learning_rate, amsgrad=args.amsgrad)
    elif args.optimizer == 'sgd':
        opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adadelta':
        opt = Adadelta(lr=args.learning_rate, amsgrad=args.amsgrad)
    else:
        assert False

    if args.freeze_classifier:
        for layer in model.layers:
            if isinstance(layer, Model):
                print("Freezing weights for classifier {}".format(layer.name))
                print(layer)
                for classifier_layer in layer.layers:
                    classifier_layer.trainable = False

    loss = { 'predictions' : args.loss} 

    model.compile(optimizer=opt, 
        loss=loss, 
        metrics={ 'predictions': ['categorical_accuracy']},
        )

    metric  = "-val_acc{val_categorical_accuracy:.6f}"
    monitor = "val_categorical_accuracy"

    save_checkpoint = ModelCheckpoint(
            join(MODEL_FOLDER, model_name+"-epoch{epoch:03d}"+metric+".hdf5"),
            monitor=monitor,
            verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=5, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')
    
    clr = CyclicLR(base_lr=args.learning_rate*4, max_lr=args.learning_rate,
                        step_size=int(math.ceil(len(ids_train)  / args.batch_size)) * 1, mode='exp_range',
                        gamma=0.99994)

    accuracy_callback = AccuracyReset(join(MODEL_FOLDER, model_name+"-epoch{epoch:03d}-group{group:03d}.hdf5"))
    callbacks = [save_checkpoint, accuracy_callback]

    if args.cyclic_learning_rate:
        callbacks.append(clr)
    else:
        callbacks.append(reduce_lr)
    
    # an epoch is just number of training samples, however if using class-aware sampling items are 
    # oversampled so one epoch does not see all distinct training items.
    model.fit_generator(
            generator        = gen(ids_train, args.batch_size, accuracy_callback = accuracy_callback),
            steps_per_epoch  = int(math.ceil(len(ids_train)  / args.batch_size)),
            validation_data  = gen(ids_val, args.batch_size, training = False),
            validation_steps = int(math.ceil(len(ids_val) / args.batch_size)),
            epochs = args.max_epoch,
            callbacks = callbacks,
            initial_epoch = last_epoch,
            class_weight={  'predictions': class_weight } if not args.class_aware_sampling else None)

elif args.test or args.test_train:

    # build model with two outputs, regular one plus unscaled logits from hadamard classifier
    # FIX: this will fail if not using hadamard
    model = Model(inputs=model.input, outputs=model.outputs + [model.get_layer('logits').output[1]])
    model.summary()

    if args.test:
        with open(TEST_CSV, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)
            all_test_ids = [ ]
            for row in reader:
                all_test_ids.append(row[0][1:-1])

    csv_name  = Path('csv') / (os.path.splitext(os.path.basename(args.model if args.model else args.weights))[0] +
      ('_test' if args.test else '_train') + '.csv')

    if args.test:
        all_ids  = all_test_ids
        jpgs_dir = TEST_DIR
        results  = defaultdict(dict)
    else:
        all_ids  = list(TRAIN_IDS)[:20000] # CHANGE
        jpgs_dir = TRAIN_DIR
        results  = defaultdict(dict)

    with Pool(min(args.batch_size, cpu_count())) as pool:
        process_item_func  = partial(process_item, predict = True)

        with open(csv_name, 'w') as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['id','landmarks'])

            imgs = np.empty((args.batch_size, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

            batch_id = 0
            batch_idx = [ ]

            def predict_minibatch():
                predictions, logits = model.predict(imgs[:batch_id])
                cats = np.argmax(predictions, axis=1)
                for i, (cat, logit, _idx) in enumerate(zip(cats, logits, batch_idx)):
                    score = predictions[i, cat]
                    landmark = cat_to_landmark[cat]
                    if results is not None:
                        results[landmark][idx] = logit
                    #np.save(Path('logits') / idx, logit)
                    if (score >= args.threshold) and (landmark != -1):
                        csv_writer.writerow([_idx, "{} {}".format(landmark, score)])
                    else:
                        csv_writer.writerow([_idx, ""])    

            for idxs in tqdm(
                (all_ids[ii:ii+args.batch_size] for ii in range(0, len(all_ids), args.batch_size)), 
                total=math.ceil(len(all_ids) / args.batch_size)):

                items = [Path(jpgs_dir) / (idx + '.jpg') for idx in idxs]

                batch_results = pool.map(process_item_func, items)

                for idx, (img, _, _) in zip(idxs, batch_results):

                    if img is not None:

                        imgs[batch_id,...] = img
                        batch_idx.append(idx)
                        batch_id += 1

                        if batch_id == args.batch_size:
                            predict_minibatch()
                            batch_id = 0
                            batch_idx = [ ]
                    else:
                        csv_writer.writerow([idx, ""])

            # predict remaining items (if any)
            if batch_id != 0:
                predict_minibatch()

    if results is not None:
        for landmark, dict_idx_logits in tqdm(results.items()):
            np.savez(Path('logits') / str(landmark), **dict_idx_logits)



