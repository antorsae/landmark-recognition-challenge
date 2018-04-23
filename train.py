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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, SeparableConv2D
from keras.utils import to_categorical
from keras.applications import *
from keras import backend as K
from keras.engine.topology import Layer
import keras.losses

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
from multiprocessing import cpu_count, Process, Queue, JoinableQueue

from functools import partial
from itertools import  islice
from conditional import conditional

from collections import defaultdict
import copy

import imgaug as ia
from imgaug import augmenters as iaa

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
# TODO tf seed

parser = argparse.ArgumentParser()
# general
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-g', '--gpus', type=int, default=None, help='Number of GPUs to use')
parser.add_argument('-v', '--verbose', action='store_true', help='Pring debug/verbose info')
parser.add_argument('-b', '--batch-size', type=int, default=6, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning_rate', type=float, default=None, help='Initial learning rate')
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
parser.add_argument('-fca', '--fully-connected-activation', type=str, default='relu', help='Activation function to use in FC layers, e.g. -fca relu|selu|prelu|leakyrelu|elu|...')
parser.add_argument('-bn', '--batch-normalization', action='store_true', help='Use batch normalization in FC layers')
parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
parser.add_argument('-lkf', '--learn-kernel-filter', action='store_true', help='Add a trainable kernel filter before classifier')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
parser.add_argument('-rp', '--reduce-pooling', type=int, default=None, help='If using pooling none add conv layers to reduce features, e.g. -rp 128')

# training regime
parser.add_argument('-cs', '--crop-size', type=int, default=256, help='Crop size')
parser.add_argument('-vpc', '--val-percent', type=float, default=0.15, help='Val percent')
parser.add_argument('-cc', '--center-crops', nargs='*', type=int, default=[], help='Train on center crops only (not random crops) for the selected classes e.g. -cc 1 6 or all -cc -1')
parser.add_argument('-nf', '--no-flips', action='store_true', help='Dont use orientation flips for augmentation')
parser.add_argument('-naf', '--non-aggressive-flips', action='store_true', help='Non-aggressive flips for augmentation')
parser.add_argument('-fcm', '--freeze-classifier', action='store_true', help='Freeze classifier weights (useful to fine-tune FC layers)')
parser.add_argument('-cas', '--class-aware-sampling', action='store_true', help='Use class aware sampling to balance dataset (instead of class weights)')
parser.add_argument('-xl', '--experimental-loss', action='store_true', help='Use experimental loss to get flat class probabily distribution on predictions')
parser.add_argument('-mu', '--mix-up', action='store_true', help='Use mix-up see: https://arxiv.org/abs/1710.09412')
parser.add_argument('-gc', '--gradient-checkpointing', action='store_true', help='Enable for huge batches, see https://github.com/openai/gradient-checkpointing')

# dataset (training)
parser.add_argument('-x', '--extra-dataset', action='store_true', help='Use dataset from https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235')
parser.add_argument('-xx', '--flickr-dataset', action='store_true', help='Use Flickr CC images dataset')

# test
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-tcs', '--test-crop-supersampling', default=1, type=int, help='Factor of extra crops to sample during test, especially useful when crop size is less than 512, e.g. -tcs 4')
parser.add_argument('-tta', action='store_true', help='Enable test time augmentation')
parser.add_argument('-e', '--ensembling', type=str, default='arithmetic', help='Type of ensembling: arithmetic|geometric|argmax for TTA')
parser.add_argument('-em', '--ensemble-models', nargs='*', type=str, default=None, help='Type of ensembling: arithmetic|geometric|argmax for TTA')
parser.add_argument('-th', '--threshold', default=0.5, type=float, help='Ignore soft probabilities less than threshold, e.g. -th 0.6')

args = parser.parse_args()

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

if args.gradient_checkpointing:
    import memory_saving_gradients
    K.__dict__["gradients"] = memory_saving_gradients.gradients_speed

TRAIN_JPGS   = set(Path('train-dl').glob('*.jpg'))
TRAIN_IDS    = { os.path.splitext(os.path.basename(item))[0] for item in TRAIN_JPGS }
TEST_JPGS    = set(Path('test').glob('*.jpg'))
TEST_IDS     = { os.path.splitext(os.path.basename(item))[0] for item in TEST_JPGS  }
MODEL_FOLDER        = 'models'
CSV_FOLDER          = 'csv'
TRAIN_CSV           = 'train.csv'
TEST_CSV            = 'test.csv'

DISTRACTOR_JPGS   = list(Path('distractors').glob('*.jpg'))
DISTRACTOR_IDS    = { os.path.splitext(os.path.basename(item))[0] for item in DISTRACTOR_JPGS }

CROP_SIZE = args.crop_size

id_to_landmark  = { }
id_to_cat       = { }
id_times_seen   = { }

landmark_to_ids = defaultdict(list)
cat_to_ids      = defaultdict(list)
landmark_to_cat = { }

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
                cat += 1 
            id_to_landmark[idx] = landmark
            id_to_cat[idx]      = landmark_cat
            id_times_seen[idx]  = 0
            landmark_to_ids[landmark].append(idx)
            cat_to_ids[landmark_cat].append(idx)

# add distractors
landmark = -1
landmark_cat = cat
landmark_to_cat[landmark] = landmark_cat
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

ids_to_dup = [ids[0] for cat,ids in cat_to_ids.items() if len(ids) == 1]

print(len(ids_to_dup))

TRAIN_JPGS = list(TRAIN_JPGS) + ids_to_dup 

#n_val_distractors = int(len(TRAIN_JPGS) * args.val_percent)

TRAIN_JPGS += DISTRACTOR_JPGS#[n_val_distractors:]
TRAIN_CATS = [ get_class(idx) for idx in TRAIN_JPGS ]

print("Total items in set {} of which {:.2f}% are distractors".format(
    len(TRAIN_JPGS), 
    100. * len(DISTRACTOR_JPGS) / len(TRAIN_JPGS)))


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
            sometimes(iaa.Crop(
                percent=(0, 0.2),
            )),
            sometimes(iaa.Affine(
                scale={"x": (1, 1.2), "y": (1, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-5, 5), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode="reflect" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 1),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
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

def process_item(item, aug = False, training = False):

    np.random.seed()
    random.seed()

    load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
    load_img           = lambda img_path: np.array(Image.open(img_path))

    class_idx = get_class(item)

    validation = not training 

    try:
        img = load_img_fast_jpg(item)
    except Exception:
        try:
            img = load_img(item)
        except Exception:
            if args.verbose:
                print('Decoding error:', item)
            return None, None, item

    shape = list(img.shape[:2])

    # discard images that do not have right resolution
    #if shape not in [resolution[:2] for resolution in RESOLUTIONS[class_idx]]:
    #    return None

    # some images may not be downloaded correctly and are B/W, discard those
    if img.ndim != 3:
        if args.verbose:
            print('Ndims !=3 error:', item)
        return None, None, item

    if img.shape[2] != 3:
        if args.verbose:
            print('More than 3 channels error:', item)
        return None, None, item

    if training and aug:
        img = augment(img)
        if np.random.random() < 0.0:
            show_image(img)
    else:
        img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))

    img = preprocess_image(img)

    if args.verbose:
        print("ap: ", img.shape, item)

    one_hot_class_idx = to_categorical(class_idx, N_CLASSES)

    return img, one_hot_class_idx, item

def process_item_worker(jobs, results):
    while True:
        task = jobs.get()
        item, aug, training = task
        results.put(process_item(item, aug, training))

def gen(items, batch_size, training=True):

    validation = not training 

    # X image crops
    X = np.empty((batch_size, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

    # class index
    y = np.empty((batch_size, N_CLASSES),               dtype=np.float32)
    
    if args.class_aware_sampling:
        items_per_class = defaultdict(list)
        for item in items:
            class_idx = get_class(item)
            items_per_class[class_idx].append(item)

        items_per_class_running=copy.deepcopy(items_per_class)
        classes = list(range(N_CLASSES))
        classes_running_copy = [ ]

    #p = Pool(min(args.batch_size, cpu_count()))
    process_item_func  = partial(process_item, training=training)

    jobs    = Queue(args.batch_size * 4)
    results = JoinableQueue(args.batch_size * 2)

    [Process(target=process_item_worker, args=(jobs, results)).start() for _ in range(cpu_count() - 1)]

    bad_items = set()
    i = 0

    while True:

        if training and not args.class_aware_sampling:
            random.shuffle(items)

        batch_idx = 0

        items_done  = 0
        while items_done < len(items):
            while not jobs.full():
                if training and args.class_aware_sampling:
                    if len(classes_running_copy) == 0:
                        random.shuffle(classes)
                        classes_running_copy = copy.copy(classes)
                    random_class = classes_running_copy.pop()
                    if len(items_per_class_running[random_class]) == 0:
                        random.shuffle(items_per_class_running[random_class])
                        items_per_class_running[random_class]=copy.deepcopy(items_per_class[random_class])
                    item = items_per_class_running[random_class].pop()
                else:
                    item = items[i % len(items)]
                    i += 1
                aug = False if id_times_seen[get_id(item)] == 0 else True
                jobs.put((item, aug, training))
                id_times_seen[get_id(item)] += 1
                items_done += 1

            get_more_results = True
            while get_more_results:
                _X, _y, _item = results.get() # blocks if none
                results.task_done()

                if _X is not None:
                    X[batch_idx], y[batch_idx] = _X, _y
                    batch_idx += 1
                    if batch_idx == batch_size:
                        yield(X, y)
                        batch_idx = 0
                else: # if batch result is None
                    bad_items.add(_item)

                get_more_results = not results.empty()

        if len(bad_items) > 0:
            print("\nRejected {} items: {}".format('trainining' if training else 'validation', len(bad_items)))

# MAIN
if args.model:
    print("Loading model " + args.model)

    model = load_model(args.model, compile=False if args.test or (args.learning_rate is not None) else True)
    # e.g. DenseNet201_do0.3_doc0.0_avg-epoch128-val_acc0.964744.hdf5
    match = re.search(r'(([a-zA-Z\d]+)_cs[,A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    args.classifier = match.group(2)
    CROP_SIZE = args.crop_size  = model.get_input_shape_at(0)[0][1]
    print("Overriding classifier: {} and crop size: {}".format(args.classifier, args.crop_size))
    last_epoch = int(match.group(3))
    if args.learning_rate == None and not args.test:
        dummy_model = model
        args.learning_rate = K.eval(model.optimizer.lr)
        print("Resuming with learning rate: {:.2e}".format(args.learning_rate))

    predictions_name = model.outputs[0].name

elif not args.ensemble_models:
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

    trainable = False
    n_trainable = 0
    for i, layer in enumerate(classifier_model.layers):
        if i >= args.freeze:
            trainable = True
            n_trainable += 1
        layer.trainable = trainable

    print("Base model has " + str(n_trainable) + "/" + str(len(classifier_model.layers)) + " trainable layers")

    x = input_image

    x = classifier_model(x)

    if args.reduce_pooling and x.shape.ndims == 4:

        pool_features = int(x.shape[3])

        for it in range(int(math.log2(pool_features/args.reduce_pooling))):

            pool_features //= 2
            x = Conv2D(pool_features, (3, 3), padding='same', use_bias=False, name='reduce_pooling{}'.format(it))(x)
            x = BatchNormalization(name='bn_reduce_pooling{}'.format(it))(x)
            x = Activation('relu', name='relu_reduce_pooling{}'.format(it))(x)
        
    if x.shape.ndims > 2:
        x = Reshape((-1,), name='reshape0')(x)

    if args.dropout_classifier != 0.:
        x = Dropout(args.dropout_classifier, name='dropout_classifier')(x)

    if not args.no_fcs:
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

    prediction   = Dense(N_CLASSES, activation ="softmax", name= "predictions")(x)

    model = Model(inputs=(input_image), outputs=(prediction))

    model_name = args.classifier + \
        '_cs{}'.format(args.crop_size) + \
        ('_fc{}'.format(','.join([str(fc) for fc in args.fully_connected_layers])) if not args.no_fcs else '_nofc') + \
        ('_bn' if args.batch_normalization else '') + \
        ('_kf' if args.kernel_filter else '') + \
        ('_lkf' if args.learn_kernel_filter else '') + \
        '_doc' + str(args.dropout_classifier) + \
        '_do'  + str(args.dropout) + \
        '_dol' + str(args.dropout_last) + \
        '_' + args.pooling + \
        ('_x' if args.extra_dataset else '') + \
        ('_xx' if args.flickr_dataset else '') + \
        ('_cc{}'.format(','.join([str(c) for c in args.center_crops])) if args.center_crops else '') + \
        ('_nf' if args.no_flips else '') + \
        ('_naf' if args.non_aggressive_flips else '') + \
        ('_cas' if args.class_aware_sampling else '') + \
        ('_mu' if args.mix_up else '') 

    print("Model name: " + model_name)

    if args.weights:
            model.load_weights(args.weights, by_name=True, skip_mismatch=True)
            match = re.search(r'([,A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.weights)
            last_epoch = int(match.group(2))

def print_distribution(ids, classes=None, prediction_probabilities=None):
    if classes is None:
        classes = [get_class(idx) for idx in ids]
    classes=np.array(classes)
    classes_count = np.bincount(classes)
    threshold = 0.7
    poor_prediction_probabilities = 0
    for class_idx, (class_name, class_count) in enumerate(zip(CLASSES, classes_count)):
        if prediction_probabilities is not None:
            prediction_probabilities_this_class = prediction_probabilities[classes == class_idx, class_idx]
            poor_prediction_probabilities_this_class = (prediction_probabilities_this_class < threshold ).sum()
            poor_prediction_probabilities += poor_prediction_probabilities_this_class
            poor_prediction_probabilities_this_class /= prediction_probabilities_this_class.size
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)) + \
            (' Poor predictions: {:04.1f}%'.format(100 * poor_prediction_probabilities_this_class) if prediction_probabilities is not None else ''))
    if prediction_probabilities is not None:
        print("                                Total poor predictions: {:04.1f}% (threshold = {:03.1f})".format( \
            100. * poor_prediction_probabilities / classes.size, threshold))

def save_csv_and_npy(ids, prediction_probabilities, classes, filename_preffix, save_npy=True):
    prediction_probabilities = np.squeeze(np.array(prediction_probabilities))
    print("Test set predictions distribution:")
    print_distribution(None, classes=classes, prediction_probabilities=prediction_probabilities)
    print("Predictions as per old-school model inference:")
    print("kg submit {}".format(filename_preffix + '.csv'))

    if save_npy:
        np.save(filename_preffix + '.npy', prediction_probabilities)

    items_per_class = len(prediction_probabilities) // N_CLASSES # it works b/c test dataset length is divisible by N_CLASSES
    csv_name  = filename_preffix + '_by_probability.csv'
    poor_predictions = 0
    with open(csv_name, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['fname','camera'])
        sum_prediction_probabilities_by_class = np.sum(prediction_probabilities, axis=(0,))
        for class_idx in np.argsort(sum_prediction_probabilities_by_class)[::-1]:
            largest_idx = np.argpartition(prediction_probabilities[:,class_idx], -items_per_class)[-items_per_class:]
            prediction_probabilities_sum = np.sum(prediction_probabilities[largest_idx], axis=1)
            prediction_probabilities_sum_zeros = prediction_probabilities_sum.size - np.count_nonzero(prediction_probabilities_sum)
            poor_predictions += np.sum((prediction_probabilities_sum > 1.) | (prediction_probabilities_sum < 0.7))
            prediction_probabilities[largest_idx] = -np.inf
            ids_by_class = [ids[largest_id] for largest_id in largest_idx]
            for largest_id in ids_by_class:
                csv_writer.writerow([largest_id.split('/')[-1], CLASSES[class_idx]])
    print("Poor predictions: {}".format(poor_predictions) )
    print("Predictions assuming prior flat probability distribution on test dataset:")
    print("kg submit {}".format(csv_name))
    csvfile.close()

if not args.ensemble_models:
    model.summary()
    model = multi_gpu_model(model, gpus=args.gpus)

if not (args.test or args.test_train or args.ensemble_models):

    # TRAINING
    ids_train, ids_val, _, _ = train_test_split(TRAIN_JPGS, TRAIN_CATS, test_size=args.val_percent, random_state=SEED, stratify=TRAIN_CATS)

    #ids_val += DISTRACTOR_JPGS[:n_val_distractors]

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

    def calculate_mAP(y_true,y_pred):
        num_classes = y_true.shape[1]
        average_precisions = []
        relevant = K.sum(K.round(K.clip(y_true, 0, 1)))
        tp_whole = K.round(K.clip(y_true * y_pred, 0, 1))
        for index in range(num_classes):
            temp = K.sum(tp_whole[:,:index+1],axis=1)
            average_precisions.append(temp * (1/(index + 1)))
        AP = Add()(average_precisions) / relevant
        mAP = K.mean(AP,axis=0)
        return mAP

    # TODO: implement this correctly.
    def weighted_loss(weights):
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
        return loss

    def categorical_crossentropy_and_variance(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred) + 10 * K.var(K.mean(y_pred, axis=0))

    if args.freeze_classifier:
        for layer in model.layers:
            if isinstance(layer, Model):
                print("Freezing weights for classifier {}".format(layer.name))
                print(layer)
                for classifier_layer in layer.layers:
                    classifier_layer.trainable = False

    loss = { 'predictions' : 'categorical_crossentropy'} 

    # monkey-patch loss so model loads ok
    # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
    #keras.losses.categorical_crossentropy_and_variance = categorical_crossentropy_and_variance
    keras.metrics.calculate_mAP = calculate_mAP

    model.compile(optimizer=opt, 
        loss=loss, 
        metrics={ 'predictions': ['accuracy']},
        )

    metric  = "-val_acc{val_" +  "acc:.6f}"
    monitor = "val_" +  "acc"

    save_checkpoint = ModelCheckpoint(
            join(MODEL_FOLDER, model_name+"-epoch{epoch:03d}"+metric+".hdf5"),
            monitor=monitor,
            verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=5, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')
    
    if False:
        clr = CyclicLR(base_lr=args.learning_rate, max_lr=args.learning_rate*10,
                            step_size=int(math.ceil(len(ids_train)  // args.batch_size)) * 4, mode='exp_range',
                            gamma=0.99994)

    callbacks = [save_checkpoint]

    if args.cyclic_learning_rate:
        callbacks.append(clr)
    else:
        callbacks.append(reduce_lr)
    
    model.fit_generator(
            generator        = gen(ids_train, args.batch_size),
            steps_per_epoch  = int(math.ceil(len(ids_train)  // args.batch_size)),
            validation_data  = gen(ids_val, args.batch_size, training = False),
            validation_steps = math.ceil(len(ids_val) // args.batch_size),
            epochs = args.max_epoch,
            callbacks = callbacks,
            initial_epoch = last_epoch,
            class_weight={  'predictions': class_weight } if not args.class_aware_sampling else None)

elif args.test or args.test_train:
    # TEST
    if args.test:
        ids = glob.glob(join(TEST_FOLDER,'*.tif'))
    elif args.test_train:
        ids = glob.glob(join(TRAIN_FOLDER,'*/*.jpg'))
    else:
        assert False

    ids.sort()

    TTA_TRANSFORMS = [[], 
                    ['orientation_1'],
                    ['orientation_2'],
                    ['orientation_3'],
                    ['manipulation_jpg70'],
                    ['manipulation_jpg90'],
                    ['manipulation_gamma0.8'],
                    ['manipulation_gamma1.2'],
                    ['manipulation_bicubic1.5'],
                    ['manipulation_bicubic2.0'],
                    ['orientation_1', 'manipulation_jpg70'],
                    ['orientation_1', 'manipulation_jpg90'],
                    ['orientation_1', 'manipulation_gamma0.8'],
                    ['orientation_1', 'manipulation_gamma1.2'],
                    ['orientation_1', 'manipulation_bicubic1.5'],
                    ['orientation_1', 'manipulation_bicubic2.0'],
                    ['orientation_2', 'manipulation_jpg70'],
                    ['orientation_2', 'manipulation_jpg90'],
                    ['orientation_2', 'manipulation_gamma0.8'],
                    ['orientation_2', 'manipulation_gamma1.2'],
                    ['orientation_2', 'manipulation_bicubic1.5'],
                    ['orientation_2', 'manipulation_bicubic2.0'],
                    ['orientation_3', 'manipulation_jpg70'],
                    ['orientation_3', 'manipulation_jpg90'],
                    ['orientation_3', 'manipulation_gamma0.8'],
                    ['orientation_3', 'manipulation_gamma1.2'],
                    ['orientation_3', 'manipulation_bicubic1.5'],
                    ['orientation_3', 'manipulation_bicubic2.0'],
                    ]
#    '''
    match = re.search(r'([^/]*)\.hdf5', args.model)
    model_name = match.group(1) + ('_tta' + str(len(TTA_TRANSFORMS)) + '_' + args.ensembling if args.tta else '')
    csv_name   = os.path.join(CSV_FOLDER, 'submission_' + model_name + '.csv')
    with conditional(args.test, open(csv_name, 'w')) as csvfile:

        if args.test:
            csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['fname','camera'])
            classes = []
            prediction_probabilities = []
        else:
            correct_predictions = 0

        for i, idx in enumerate(tqdm(ids)):

            img = np.array(Image.open(idx))

            if args.test_train:
                img, _ = get_crop(img, 512*2, random_crop=False)

            original_img = img

            original_manipulated = np.float32([1. if idx.find('manip') != -1 else 0.])

            if args.test and args.tta:
                transforms_list = TTA_TRANSFORMS
            elif args.test_train:
                transforms_list = [[], ['orientation'], ['manipulation'], ['manipulation', 'orientation']]
            else:
                transforms_list = [[]]

            ssx = int(img.shape[1] / CROP_SIZE * args.test_crop_supersampling)
            ssy = int(img.shape[0] / CROP_SIZE * args.test_crop_supersampling)

            img_batch         = np.zeros((len(transforms_list)* ssx * ssy, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
            manipulated_batch = np.zeros((len(transforms_list)* ssx * ssy, 1),  dtype=np.float32)
            center_crop_batch = np.zeros((len(transforms_list)* ssx * ssy, 2),  dtype=np.float32)

            batch_idx = 0
            for transforms in transforms_list:
                img = np.copy(original_img)
                manipulated = np.copy(original_manipulated)

                orientation_transform  = [m for m in transforms if m.startswith('orientation')]
                manipulation_transform = [m for m in transforms if m.startswith('manipulation')]
                if orientation_transform:
                    assert len(orientation_transform) == 1
                    rot_times = orientation_transform[0][12:]
                    rotations = int(rot_times) if rot_times != '' else random.choice([1,3] if args.non_aggressive_flips else [1,2,3])
                    img = np.rot90(img, rotations, (0,1))
                if manipulation_transform:
                    assert len(manipulation_transform) == 1
                    if original_manipulated:
                        continue
                    else:
                        manipulation = manipulation_transform[0][13:]
                        if manipulation == '':
                            manipulation = random_choice('jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic1.5', 'bicubic2.0')
                            # cannot do  'bicubic0.5', 'bicubic0.8' b/c resulting image is too small

                        img, manipulation_idx = get_random_manipulation(img, manipulation=manipulation)
                        if manipulation.startswith('bicubic'):
                            img, _ = get_crop(img, 512, random_crop=False)

                        manipulated = np.float32([1.])

                if args.test_train:
                    img, _ = get_crop(img, 512, random_crop=False)

                sx = img.shape[1] / CROP_SIZE
                sy = img.shape[0] / CROP_SIZE

                for x in np.linspace(0, img.shape[1] - CROP_SIZE, args.test_crop_supersampling * sx, dtype=np.int64):
                    for y in np.linspace(0, img.shape[0] - CROP_SIZE, args.test_crop_supersampling * sy, dtype=np.int64):
                        _img = np.copy(img[y:y+CROP_SIZE, x:x+CROP_SIZE])
                        img_batch[batch_idx]         = preprocess_image(_img)
                        manipulated_batch[batch_idx] = manipulated
                        batch_idx += 1
                        # TODO: For crop size < 512 make a decent approximation of center_crop_batch

            if len(model.inputs)== 1:
                _inputs = img_batch[:batch_idx]
            elif len(model.inputs) == 2:
                _inputs = [img_batch[:batch_idx],manipulated_batch[:batch_idx]]
            else:
                _inputs = [img_batch[:batch_idx],manipulated_batch[:batch_idx], center_crop_batch[:batch_idx]]

            _output = model.predict_on_batch(_inputs)

            if len(model.outputs) == 1:
                prediction = _output
            else:
                prediction = _output[0]

            if prediction.shape[0] != 1: # TTA
                if args.ensembling == 'geometric':
                    prediction = np.log(prediction + K.epsilon()) # avoid numerical instability log(0)
                    prediction = np.mean(prediction, axis=0, keepdims=True)
                    prediction = np.exp(prediction) - K.epsilon() # get soft-probs again so we can see poor predictions
                elif args.ensembling == 'argmax':
                    prediction = prediction[np.unravel_index(np.argmax(prediction), prediction.shape)[:-1]]
                    prediction = np.expand_dims(prediction, axis=0)
                elif args.ensembling == 'arithmetic':
                    prediction = np.mean(prediction, axis=0, keepdims=True)
                else:
                    Print("Unknown ensembling type")
                    assert False

            prediction_class_idx = np.argmax(prediction)

            if args.test_train:
                class_idx = get_class(idx.split('/')[-2])
                if class_idx == prediction_class_idx:
                    correct_predictions += 1

            if args.test:
                csv_writer.writerow([idx.split('/')[-1], CLASSES[prediction_class_idx]])
                classes.append(prediction_class_idx)
                prediction_probabilities.append(prediction)

        if args.test_train:
            print("Accuracy: " + str(correct_predictions / (len(transforms) * i)))

        if args.test:
            save_csv_and_npy(ids, prediction_probabilities, classes, csv_name[:-4])

elif args.ensemble_models:
    # will build array (len(args.ensemble_models), 2640, 10)
    TEST_ITEMS = 2640
    predictions = np.zeros((len(args.ensemble_models), TEST_ITEMS, len(CLASSES)))
    filename_preffix = os.path.join(CSV_FOLDER, 'ensemble_' + str(len(args.ensemble_models)) + '_' + args.ensembling + '_th' + str(args.threshold) )
    for it, predictions_path in enumerate(args.ensemble_models):
        preds_single_model = np.load(predictions_path) 
        classes_single_model = np.squeeze(np.argmax(preds_single_model, axis=1))
        print("Test set predictions distribution for model: " + predictions_path)
        print_distribution(None, classes=classes_single_model, prediction_probabilities=preds_single_model)
        print()
        assert preds_single_model.shape[:2] == (TEST_ITEMS,len(CLASSES))
        predictions[it,...] = preds_single_model
        # e.g. DenseNet201_do0.3_doc0.0_avg-epoch128-val_acc0.964744.hdf5
        print(predictions_path)
        match = re.search(r'(([a-zA-Z\d]+)_cs[,A-Za-z_\d\.]+)-epoch(\d+)-val_acc([0-9\.]+).*\.npy', predictions_path)
        filename_preffix += '_' + match.group(4)
    
    ids = glob.glob(join(TEST_FOLDER,'*.tif'))
    ids.sort()
    
    print("Filename preffix: " + filename_preffix)
    predictions[predictions < args.threshold] = np.nan

    if args.ensembling == 'geometric':
        predictions = np.log(predictions + K.epsilon()) # avoid numerical instability log(0)
        predictions = np.nanmean(predictions, axis=0, keepdims=True)
        predictions = np.exp(predictions) - K.epsilon() # get soft-probs again so we can see poor predictions
    elif args.ensembling == 'argmax':
        predictions = predictions[np.unravel_index(np.argmax(predictions), predictions.shape)[:-1]]
        predictions = np.expand_dims(predictions, axis=0)
    elif args.ensembling == 'arithmetic':
        predictions = np.nanmean(predictions, axis=0, keepdims=True)
    else:
        Print("Unknown ensembling type")
        assert False

    predictions = np.nan_to_num(predictions)

    classes = np.squeeze(np.nanargmax(predictions, axis=2))


    save_csv_and_npy(ids, predictions, classes, filename_preffix, save_npy=False)



