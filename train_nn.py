import argparse
import glob
import random
import pickle
import os
from os.path import join
from pathlib import Path
import numpy as np
import math
import re

from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, AveragePooling2D, Reshape, SeparableConv2D
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import to_categorical
import keras.losses
from multi_gpu_keras import multi_gpu_model
from clr_callback import CyclicLR
from hadamard import HadamardClassifier
import re
import itertools
from statistics import mean

from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# yapf: disable
parser = argparse.ArgumentParser()
# general
parser.add_argument('--max-epoch', type=int, default=1000, help='Epoch to run')
parser.add_argument('-g', '--gpus', type=int, default=None, help='Number of GPUs to use')
parser.add_argument('-b', '--batch-size', type=int, default=48, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning-rate', type=float, default=None, help='Initial learning rate')
parser.add_argument('-clr', '--cyclic_learning_rate', action='store_true', help='Use cyclic learning rate https://arxiv.org/abs/1506.01186')
parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer to use in training -o adam|sgd|adadelta')
parser.add_argument('--amsgrad', action='store_true', help='Apply the AMSGrad variant of adam|adadelta from the paper "On the Convergence of Adam and Beyond".')
parser.add_argument('-nns', '--networks', nargs='+', type=str, default=['InceptionV3_cs299', 'NASNetLarge_cs331', 'VGG16Places365_cs256', 'VGG16PlacesHybrid1365_cs256', 'Xception_cs299', 'ResNet50_cs224', 'DenseNet201_cs224'], help='NNs models')
parser.add_argument('-cache', '--usecache', action='store_true', help='Use cached train data')
parser.add_argument('-tlp', '--top-landmark-percent', type=float, default=1., help='Use only top percent of landmarks, e.g -tlp 0.1 uses 10% of landmarks')

# architecture/model
parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
parser.add_argument('-lo', '--loss', type=str, default='categorical_crossentropy', help='Loss function')
parser.add_argument('-do', '--dropout', type=float, default=0., help='Dropout rate for first FC layer')
parser.add_argument('-dol', '--dropout-last', type=float, default=0., help='Dropout rate for last FC layer')
parser.add_argument('-fc', '--fully-connected-layers', nargs='+', type=int, default=[512, 256], help='Specify FC layers, e.g. -fc 1024 512 256')
parser.add_argument('-bn', '--batch-normalization', action='store_true', help='Use batch normalization in FC layers')
parser.add_argument('-fca', '--fully-connected-activation', type=str, default='relu', help='Activation function to use in FC layers, e.g. -fca relu|selu|prelu|leakyrelu|elu|...')
parser.add_argument('-hp', '--hadamard', action='store_true', help='Use Hadamard projection instead of FC layers, see https://arxiv.org/pdf/1801.04540.pdf')
parser.add_argument('-tn', '--top-n', type=int, default=16, help='Use top-N NNs)')
parser.add_argument('-d', '--dense', action='store_true', help='Use dense model')

# test
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV/npy submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
# yapf: enable

args = parser.parse_args()

training = not (args.test or args.test_train)
n_nets = len(args.networks)

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if args.gpus is None:
    args.gpus = len(get_available_gpus())

args.batch_size *= max(args.gpus, 1)

# data
N_CLASSES = 14951
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

distances=[]
landmarks=[]
testids  =None

for n, network in enumerate(tqdm(args.networks)):
    #vector = np.zeros((N_CLASSES), dtype=np.float32)
    #print(network)
    distances.append(np.lib.format.open_memmap('results/%s.distances_tk64_train.npy' % network, mode='c'))
    landmarks.append(np.lib.format.open_memmap('results/%s.landmarks_tk64_train.npy' % network, mode='c'))
    _testids = pickle.load(open('results/%s.testids_train' % network, 'rb'))
    if testids is None:
        testids = _testids
    else:
        assert testids == _testids

if args.dense:
    # normalize distances
    for d in distances:
        max_d = d.max()
        d /= max_d
elif False:
    for dd,ll in tqdm(zip(distances, landmarks)):
        for d,l in tqdm(zip(dd,ll)):
            ul, lc = np.unique(l, return_counts=True)
            for dl in ul[lc>1]:
                d[dl==l] = d[dl==l].sum()


testids = [int(t) for t in testids]
assert len(set(testids)) == max(testids)+1

_landmarks, _landmark_counts = np.unique(testids, return_counts=True)

_landmark_counts_order = np.argsort(_landmark_counts)[::-1]

_landmark_counts_cumsum = np.cumsum(_landmark_counts[_landmark_counts_order])

print("{} @ 10% {} @ 25% {} @ 50% {} 75% {} @ 100%".format(
    _landmark_counts_cumsum[  N_CLASSES//10], 
    _landmark_counts_cumsum[  N_CLASSES//4], 
    _landmark_counts_cumsum[  N_CLASSES//2],
    _landmark_counts_cumsum[3*N_CLASSES//4],
    _landmark_counts_cumsum[-1]))

landmark_log_counts = { k : 1+ np.log(v) for k,v in enumerate(_landmark_counts)}

landmark_log_counts[-1] = 0

if args.top_landmark_percent != 1.:
    items = [i for i in range(len(testids)) if testids[i] in _landmarks[_landmark_counts_order][:int(N_CLASSES * args.top_landmark_percent)]]
    print('Using {} samples of total {} ({:.2f}%)'.format(
        len(items),
        len(testids),
        100 * len(items) / len(testids)))
else:
    items = range(len(testids))

IDX_TRAIN_SPLIT, IDX_VALID_SPLIT = train_test_split(items, test_size=0.1, random_state=SEED)

print('Training on {} samples'.format(len(IDX_TRAIN_SPLIT)))
print('Validating on {} samples'.format(len(IDX_VALID_SPLIT)))

def collate_landmarks_distances(landmarks, distances):
    u_landmarks     = []
    u_min_distances = []
    u_counts        = []
    for l,d in zip(landmarks, distances):
        u_landmark, ii, u_count = np.unique(l, return_index=True, return_counts=True)
        n_u_landmark = u_landmark.shape[0]
        u_min_distance = np.empty((n_u_landmark,), dtype=np.float32)
        u_min_distance[u_count==1] = d[ii[u_count==1]] # copy the ones with one 1 count
        for u_index, u_landmark_match in zip(np.argwhere(u_count!=1), u_landmark[u_count!=1]):
            u_min_distance[u_index] = d[l == u_landmark_match].min()
        u_landmarks.append(u_landmark)
        u_min_distances.append(u_min_distance)
        u_counts.append(u_count)
    return u_landmarks, u_min_distances, u_counts

def build_dense_vector(landmarks, distances, topN = 0):
    # landmarks is LIST of N x (topK+1; int64)
    # distances is LIST of N x (topK+1; float32)
    nets = len(landmarks)
    assert nets == len(distances)
    # u_landmarks    will be a LIST of N x (unique_landmarks; int)     each element landmark id
    # u_min_istances will be a LIST of N x (unique_landmarks; float32) each element min distance of landmarks
    # u_counts       will be a LIST of N x (unique_landmarks; int)     each element number of landmarks 
    u_landmarks, u_min_distances, u_counts = collate_landmarks_distances(landmarks, distances)
    a_landmarks, a_counts = np.unique(list(itertools.chain(*u_landmarks)), return_counts=True)
    n = a_landmarks.shape[0]
    a_min_distances = np.empty((n,), dtype=np.float32)
    # copy distances from each item in u_min_distances
    for i in range(nets):
        a_min_distances[np.isin( a_landmarks, u_landmarks[i])] = u_min_distances[i]
    # overwrite those which have >1 count with its mean
    a_min_distances[a_counts!=1] = [np.min([u_min_distances[i][u_landmarks[i]==la] 
        for i in range(nets) if u_min_distances[i][u_landmarks[i]==la].size > 0]) for la in a_landmarks[a_counts!=1]]
    # sort by avg distance of combined landmarks
    sorted_idx = np.argsort(a_min_distances)

    # wild ride!
    #random.shuffle(sorted_idx)

    # sorted landmarks
    s_landmarks = a_landmarks[sorted_idx]
    min_distances = np.zeros((nets, n), dtype=np.float32)
    counts        = np.zeros((nets, n), dtype=np.int32)
    for i in range(nets):
        min_distances[i, np.isin( a_landmarks, u_landmarks[i])] = u_min_distances[i]
        counts[i, np.isin( a_landmarks, u_landmarks[i])]        = u_counts[i]
    if topN != 0:
        _min_distances = np.zeros((nets, topN), dtype=np.float32)
        _counts        = np.zeros((nets, topN), dtype=np.int32)
        _s_landmarks   = np.zeros((topN,), dtype=np.int32)
        
        _s_landmarks[:] = -1

        _min_distances[:,:min(n,topN)] = min_distances[:,:min(n,topN)]
        _counts       [:,:min(n,topN)] = counts       [:,:min(n,topN)]
        _s_landmarks  [:min(n,topN)]   = s_landmarks  [:min(n,topN)]
        min_distances = _min_distances
        counts        = _counts
        s_landmarks   = _s_landmarks
    return min_distances, counts, s_landmarks

def dense_generator(args, IDX, train=True):
    if train:
        random.shuffle(IDX)

    x_batch = np.empty((args.batch_size, n_nets, args.top_n, 3), dtype=np.float32)
    y_batch = np.empty((args.batch_size, args.top_n+1),          dtype=np.float32)

    i  = 0
    ib = 0
    while True:
        l = []
        d = []
        idx = IDX[i]
        for net in range(n_nets):
            l.append(landmarks[net][idx][1:])
            d.append(distances[net][idx][1:])
        min_distances, counts, s_landmarks = build_dense_vector(l,d, topN=args.top_n)

        landmark_gt = int(testids[idx])
        if int(landmark_gt) in s_landmarks:
            gt = int(np.argwhere(s_landmarks==landmark_gt))
        else:
            gt = args.top_n
        x_batch[ib,..., 0] = min_distances
        x_batch[ib,..., 1] = counts
        x_batch[ib,..., 2] = [landmark_log_counts[l] for l in s_landmarks]
        if np.random.random() < 0.0000:
            print(x_batch[ib,..., 1])
            print(x_batch[ib,..., 2])

        y_batch[ib,...] = to_categorical(gt, args.top_n+1)
        ib += 1
        if ib == args.batch_size:
            yield(x_batch, y_batch)
            ib = 0

        i += 1
        if i == len(IDX):
            i = 0
            if train:
                random.shuffle(IDX)

def sparse_generator(args, IDX, train=True):
    if train:
        random.shuffle(IDX)

    x_batch = np.empty((args.batch_size, N_CLASSES, 1 + n_nets*2), dtype=np.float32)
    y_batch = np.empty((args.batch_size, N_CLASSES),         dtype=np.float32)

    for bb in range(args.batch_size):
        x_batch[bb, :, n_nets * 2] = [landmark_log_counts[i] for i in range(N_CLASSES)]

    i  = 0
    ib = 0
    x_zeros = np.empty((N_CLASSES, ), dtype=np.float32)
    while True:
        idx = IDX[i]

        for net in range(n_nets):
            x_zeros[:] = 0
            # 1: is to ignore the first NN match which in this case is the idx itself
            # [::-1] is to make sure the distance copied is the first one from distances[...]
            # and since they are ordered we want to place the first one (closest), not the last
            x_zeros[landmarks[net][idx,1:][::-1]] = distances[net][idx,1:][::-1]
            x_batch[ib,..., net] = x_zeros

            # we feed another vector of N_CLASSES withe ocurrences of each landmark in the current
            # with the expectation that the net will use this information
            lidx, lcnt = np.unique(landmarks[net][idx,1:], return_counts=True)

            x_zeros[:] = 0
            x_zeros[lidx] = lcnt
            x_batch[ib,..., net+n_nets] = x_zeros

            if False:
                # poor man's debugger
                print(landmarks[net][idx,1:])
                print(distances[net][idx,1:])
                x0 = x_batch[ib,..., net]
                x1 = x_batch[ib,..., net+n_nets]
                print(np.nonzero(x0))
                print(x0[np.nonzero(x0)])
                print(np.nonzero(x1))
                print(x1[np.nonzero(x1)])
                assert False

        y_batch[ib,...] = to_categorical(testids[idx], N_CLASSES)

        ib += 1
        if ib == args.batch_size:
            yield(x_batch, y_batch)
            ib = 0
        i += 1
        if i == len(IDX):
            i = 0
            if train:
                random.shuffle(IDX)

if args.model:
    print("Loading model " + args.model)
    model = load_model(
        args.model,
        compile=False
        if not training or (args.learning_rate is not None) else True)
    model_basename = os.path.splitext(os.path.basename(args.model))[0]
    model_parts = model_basename.split('-')
    model_name = '-'.join(
        [part for part in model_parts if part not in ['epoch', 'val_acc']])
    last_epoch = int(
        list(filter(lambda x: x.startswith('epoch'), model_parts))[0][5:])
    print("Last epoch: {}".format(last_epoch))
    if args.learning_rate == None and training:
        dummy_model = model
        args.learning_rate = K.eval(model.optimizer.lr)
        print("Resuming with learning rate: {:.2e}".format(args.learning_rate))

elif True:

    if args.learning_rate is None:
        args.learning_rate = 1e-4  # default LR unless told otherwise

    last_epoch = 0

    if args.dense:
        input = Input(shape=(n_nets, args.top_n, 3), name='input')
    else:
        input = Input(shape=(N_CLASSES, 1 + n_nets*2), name='input')        

    x = input

    if not args.dense:
        base_filters = (1 + n_nets*2)
        x = Conv1D(base_filters *2, 1, use_bias=True, activation='relu')(x) # multipliers
        #x = Conv1D(base_filters *4, 1, use_bias=True, activation='relu')(x) # multipliers
        #x = Conv1D(base_filters *2, 1, use_bias=True, activation='relu')(x) # multipliers
        x = Conv1D(1, 1,               use_bias=True, activation='relu')(x) # multipliers

    x = Flatten()(x)
    x = BatchNormalization(name='bn0')(x)

    dropouts = np.linspace(args.dropout, args.dropout_last,
                           len(args.fully_connected_layers))

    for i, (fc_layer, dropout) in enumerate(
            zip(args.fully_connected_layers, dropouts)):
        if args.batch_normalization:
            x = Dense(fc_layer, name='fc{}'.format(i))(x)
            x = BatchNormalization(name='bn{}'.format(i))(x)
            x = Activation(
                args.fully_connected_activation,
                name='act{}{}'.format(args.fully_connected_activation, i))(x)
        else:
            x = Dense(
                fc_layer,
                activation=args.fully_connected_activation,
                name='fc{}'.format(i))(x)
        if dropout != 0:
            x = Dropout(
                dropout, name='dropout_fc{}_{:04.2f}'.format(i, dropout))(x)

    n_outs = args.top_n+1 if args.dense else N_CLASSES

    if args.hadamard:
        x = HadamardClassifier(n_outs, name= "logits")(x)
    else:
        x = Dense(n_outs, name="logits")(x)

    prediction = Activation(activation="softmax", name="predictions")(x)

    model = Model(inputs=input, outputs=prediction)
    print(model.summary())

    if args.weights:
        print("Loading weights from {}".format(args.weights))
        model.load_weights(args.weights, by_name=True, skip_mismatch=True)
        match = re.search(r'([,A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5',
                          args.weights)
        last_epoch = int(match.group(2))

if training:
    if args.optimizer == 'adam':
        opt = Adam(lr=args.learning_rate, amsgrad=args.amsgrad)
    elif args.optimizer == 'sgd':
        opt = SGD(
            lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adadelta':
        opt = Adadelta(lr=args.learning_rate, amsgrad=args.amsgrad)
    else:
        assert False

    loss = {'predictions': args.loss}

    model = multi_gpu_model(model, gpus=args.gpus)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics={
            'predictions': ['categorical_accuracy']
        })

    model_name = 'nns'
    mode = 'max'
    metric = "-val_acc{val_categorical_accuracy:.6f}"
    monitor = "val_categorical_accuracy"

    save_checkpoint = ModelCheckpoint(
        join(MODEL_FOLDER,
             model_name + "-epoch{epoch:03d}" + metric + ".hdf5"),
        monitor=monitor,
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode=mode,
        period=1)

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.1,
        patience=5,
        min_lr=1e-9,
        epsilon=0.00001,
        verbose=1,
        mode=mode)

    clr = CyclicLR(
        base_lr=args.learning_rate / 4,
        max_lr=args.learning_rate,
        step_size=int(math.ceil(len(IDX_TRAIN_SPLIT) / args.batch_size)) * 1,
        mode='exp_range',
        gamma=0.99994)

    callbacks = [save_checkpoint]

    if args.cyclic_learning_rate:
        callbacks.append(clr)
    else:
        callbacks.append(reduce_lr)

    generator = dense_generator if args.dense else sparse_generator

    model.fit_generator(
        generator=generator(args, IDX_TRAIN_SPLIT, train=True),
        steps_per_epoch=np.ceil(
            float(len(IDX_TRAIN_SPLIT)) / float(args.batch_size)) - 1,
        epochs=args.max_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=generator(args, IDX_VALID_SPLIT, train=False),
        validation_steps=np.ceil(
            float(len(IDX_VALID_SPLIT)) / float(args.batch_size)) - 1,
        initial_epoch=last_epoch,
        use_multiprocessing = True if args.dense else False,
        workers = 32 if args.dense else False)

elif args.test or args.test_train:
    pass
