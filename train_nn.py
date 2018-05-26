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
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
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

<<<<<<< HEAD
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

# normalize distances
for d in distances:
    max_d = d.max()
    d /= max_d
testids = [int(t) for t in testids]
assert len(set(testids)) == max(testids)+1

_, _landmark_counts = np.unique(testids, return_counts=True)

landmark_counts = { k : 1+ np.log(v) for k,v in enumerate(_landmark_counts)}
=======
TRAIN_LABELS = None
if not args.usecache:
    TRAIN_DATA = defaultdict(lambda: np.zeros((N_CLASSES), dtype=np.float32))
    TRAIN_LABELS = dict()
    HAS_RIGHT_ANSWER = defaultdict(bool)

    for network in tqdm(args.networks):
        vector = np.zeros((N_CLASSES), dtype=np.float32)
        print(network)
        distances = np.load('results/%s.distances_tk64_train.npy' % network)
        landmarks = np.load('results/%s.landmarks_tk64_train.npy' % network)
        testids = pickle.load(open('results/%s.testids_tk64_train' % network, 'rb'))

        i = 0
        dlt = zip(distances, landmarks, testids)
        for (distance, landmark, test_id) in dlt:
            TRAIN_DATA[i][landmark[1:]] += np.reciprocal(
                distance[1:] + K.epsilon())
            #TRAIN_DATA[i][landmark[1:]] += -np.log(distance[1:] + K.epsilon())
            TRAIN_LABELS[i] = landmark[0]
            hra = landmark[0] in landmark[1:]
            HAS_RIGHT_ANSWER[i] = HAS_RIGHT_ANSWER[i] or hra
            i = i + 1
            #if i > 10:
            #    break
    TRAIN_DATA = [
        e[1] for e in sorted(TRAIN_DATA.items()) if HAS_RIGHT_ANSWER[e[0]]
    ]
    TRAIN_LABELS = [
        e[1] for e in sorted(TRAIN_LABELS.items()) if HAS_RIGHT_ANSWER[e[0]]
    ]

    # cache
    #os.makedirs('cache', exist_ok=True)
    #for part in range(math.ceil(len(TRAIN_DATA) / 100000)):
    #    np.save('cache/nns_train_data_%02d.npy' % part,
    #            TRAIN_DATA[part * 100000:(part + 1) * 100000])
    #    np.save('cache/nns_train_labels_%02d.npy' % part,
    #            TRAIN_LABELS[part * 100000:(part + 1) * 100000])
    del HAS_RIGHT_ANSWER
else:
    cache_files = sorted(glob.glob('cache/nns_train_labels_*.npy'))
    for cache_file in tqdm(cache_files):
        if TRAIN_LABELS is None:
            TRAIN_LABELS = np.load(cache_file)
        else:
            TRAIN_LABELS = np.append(TRAIN_LABELS, np.load(cache_file), axis=0)
>>>>>>> 96d1181e3c1d32c5fbb59c45ec886acb9f53bf19

landmark_counts[-1] = 0

items = range(len(testids))

IDX_TRAIN_SPLIT, IDX_VALID_SPLIT = train_test_split(items, test_size=0.1, random_state=SEED)

print('Training on {} samples'.format(len(IDX_TRAIN_SPLIT)))
print('Validating on {} samples'.format(len(IDX_VALID_SPLIT)))

def collate_landmarks_distances(landmarks, distances):
    u_landmarks     = []
    u_avg_distances = []
    u_counts        = []
    for l,d in zip(landmarks, distances):
        u_landmark, ii, u_count = np.unique(l, return_index=True, return_counts=True)
        n_u_landmark = u_landmark.shape[0]
        u_avg_distance = np.empty((n_u_landmark,), dtype=np.float32)
        u_avg_distance[u_count==1] = d[ii[u_count==1]] # copy the ones with one 1 count
        for u_index, u_landmark_match in zip(np.argwhere(u_count!=1), u_landmark[u_count!=1]):
            u_avg_distance[u_index] = d[l == u_landmark_match].min()
        u_landmarks.append(u_landmark)
        u_avg_distances.append(u_avg_distance)
        u_counts.append(u_count)
    return u_landmarks, u_avg_distances, u_counts

def build_dense_vector(landmarks, distances, topN = 0):
    # landmarks is LIST of N x (topK+1; int64)
    # distances is LIST of N x (topK+1; float32)
    nets = len(landmarks)
    assert nets == len(distances)
    # u_landmarks    will be a LIST of N x (unique_landmarks; int)     each element landmark id
    # u_avg_istances will be a LIST of N x (unique_landmarks; float32) each element avg distance of landmarks
    # u_counts       will be a LIST of N x (unique_landmarks; int)     each element number of landmarks 
    u_landmarks, u_avg_distances, u_counts = collate_landmarks_distances(landmarks, distances)
    a_landmarks, a_counts = np.unique(list(itertools.chain(*u_landmarks)), return_counts=True)
    n = a_landmarks.shape[0]
    a_avg_distances = np.empty((n,), dtype=np.float32)
    # copy distances from each item in u_avg_distances
    for i in range(nets):
        a_avg_distances[np.isin( a_landmarks, u_landmarks[i])] = u_avg_distances[i]
    # overwrite those which have >1 count with its mean
    a_avg_distances[a_counts!=1] = [np.min([u_avg_distances[i][u_landmarks[i]==la] 
        for i in range(nets) if u_avg_distances[i][u_landmarks[i]==la].size > 0]) for la in a_landmarks[a_counts!=1]]
    # sort by avg distance of combined landmarks
    sorted_idx = np.argsort(a_avg_distances)

    # wild ride!
    #random.shuffle(sorted_idx)

    # sorted landmarks
    s_landmarks = a_landmarks[sorted_idx]
    avg_distances = np.zeros((nets, n), dtype=np.float32)
    counts        = np.zeros((nets, n), dtype=np.int32)
    for i in range(nets):
        avg_distances[i, np.isin( a_landmarks, u_landmarks[i])] = u_avg_distances[i]
        counts[i, np.isin( a_landmarks, u_landmarks[i])]        = u_counts[i]
    if topN != 0:
        _avg_distances = np.zeros((nets, topN), dtype=np.float32)
        _counts        = np.zeros((nets, topN), dtype=np.int32)
        _s_landmarks   = np.zeros((topN,), dtype=np.int32)
        
        _s_landmarks[:] = -1

        _avg_distances[:,:min(n,topN)] = avg_distances[:,:min(n,topN)]
        _counts       [:,:min(n,topN)] = counts       [:,:min(n,topN)]
        _s_landmarks  [:min(n,topN)]   = s_landmarks  [:min(n,topN)]
        avg_distances = _avg_distances
        counts        = _counts
        s_landmarks   = _s_landmarks
    return avg_distances, counts, s_landmarks

def combined_generator(args, IDX, train=True):
    if train:
        random.shuffle(IDX)

    x_batch = np.empty((args.batch_size, n_nets, args.top_n, 3), dtype=np.float32)
    y_batch = np.empty((args.batch_size, args.top_n+1),         dtype=np.float32)

    i  = 0
    ib = 0
    missed = 0
    while True:
        l = []
        d = []
        idx = IDX[i]
        for net in range(n_nets):
            l.append(landmarks[net][idx][1:])
            d.append(distances[net][idx][1:])
        avg_distances, counts, s_landmarks = build_dense_vector(l,d, topN=args.top_n)

        landmark_gt = int(testids[idx])
        if int(landmark_gt) in s_landmarks:
            gt = int(np.argwhere(s_landmarks==landmark_gt))
        else:
            gt = args.top_n
        x_batch[ib,..., 0] = avg_distances
        x_batch[ib,..., 1] = counts
        x_batch[ib,..., 2] = [landmark_counts[l] for l in s_landmarks]
        if np.random.random() < 0.0000:
            print(x_batch[ib,..., 1])
            print(x_batch[ib,..., 2])

        y_batch[ib,...] = to_categorical(gt, args.top_n+1)
        ib += 1
        if ib == args.batch_size:
            yield(x_batch, y_batch)
            ib = 0
        #else:
        #    missed += 1
        i += 1
        if i == len(IDX):
            i = 0
            if train:
                random.shuffle(IDX)
            print("Missed {}".format(missed/len(IDX)))
            missed = 0


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

    input = Input(shape=(n_nets, args.top_n, 3), name='input')

    x = input
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

    if args.hadamard:
        x = HadamardClassifier(args.top_n+1, name= "logits")(x)
    else:
        x = Dense(args.top_n+1, name="logits")(x)

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

    model.fit_generator(
        generator=combined_generator(args, IDX_TRAIN_SPLIT, train=True),
        steps_per_epoch=np.ceil(
            float(len(IDX_TRAIN_SPLIT)) / float(args.batch_size)) - 1,
        epochs=args.max_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=combined_generator(args, IDX_VALID_SPLIT, train=False),
        validation_steps=np.ceil(
            float(len(IDX_VALID_SPLIT)) / float(args.batch_size)) - 1,
        initial_epoch=last_epoch,
        use_multiprocessing = True,
        workers = 31)

elif args.test or args.test_train:
    pass
