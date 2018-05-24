import glob
import numpy as np
import faiss                     # make faiss available
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import math
import os
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--features', default=512, type=int, help='Features to pick, e.g. -f 512')
parser.add_argument('-pca', '--pca', default=0, type=int, help='Use PCA to reduce dimensions, e.g. -pca 512')
parser.add_argument('-tk', '--top-k', default=16, type=int, help='Store top-k NN matches, e.g. -tk 32')
parser.add_argument('-cpu', '--cpu', action='store_true', help='Dont use GPU')
parser.add_argument('-t', '--train', action='store_true', help='Train index')
parser.add_argument('--features-dir', default='features', help='Prefix dir of computed features, index and results')
parser.add_argument('--results-dir', default='results', help='Prefix dir of computed features, index and results')
parser.add_argument('-n', '--net', default='VGG16Places365-cs256', help='Subdir of computed features, e.g. -n VGG16Places365-cs256')
parser.add_argument('-pr', '--print-results', default=0, type=int, help='Print results of the n first queries, e.g. -pr 16')
parser.add_argument('-f16', '--float16', action='store_true', help='Use float16 lookup tables')
parser.add_argument('-etnn', '--extract-train-nn', action='store_true', help='Extract train nearest neighbors instead of test ones')

args = parser.parse_args()

FEATURES_NUMBER = args.features
PCA_FEATURES    = args.pca

train = args.train
pca   = args.pca != 0
gpu   = not args.cpu

features_dir = args.features_dir + "/" + args.net

FEATURES_NPY       = features_dir + '/*.npy'
INDEX_FILENAME_PRE = args.results_dir + '/' + args.net.replace("-", "_")
INDEX_FILENAME     = INDEX_FILENAME_PRE + '.index'
INDEX_FILENAME_PK  = INDEX_FILENAME_PRE + '.pk'
INDEX_FILENAME_PCA = INDEX_FILENAME_PRE + '.pca' + str(args.pca)

res = faiss.StandardGpuResources()  # use a single GPU
co = faiss.GpuClonerOptions()
# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
if args.float16: co.useFloat16 = True

if os.path.exists(INDEX_FILENAME):
    cpu_index = faiss.read_index(INDEX_FILENAME)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co) if gpu else cpu_index

    if pca:
        mat = faiss.read_VectorTransform(INDEX_FILENAME_PCA) # todo calculate it if not there

    with open(INDEX_FILENAME_PK, 'rb') as fp:
        index_dict = pickle.load(fp)
else:
    files = sorted(glob.glob(FEATURES_NPY))
    index_dict = { }
    label_features = { }
    i = 0
    n_train_subset = 0
    for file_name in tqdm(files):
        label = file_name.split('/')[-1].split('.')[0]
        if len(label) == 16:
            continue
        features = np.load(file_name)
        assert features.shape[1] == FEATURES_NUMBER
        label_features[label] = features
        n_train_subset += max(1, features.shape[0] // 5)

    subset_i = 0
    
    if train or pca:
        train_subset = np.empty((n_train_subset, FEATURES_NUMBER), dtype=np.float32)
        
        print("Adding {} train features for training".format(n_train_subset))
        for label, features in label_features.items():
            n_features = max(1, features.shape[0] // 5)
            train_subset[subset_i:subset_i+n_features] = features[:n_features]
            #for n_feature in range(n_features):
            #    index_dict[subset_i+n_feature] = int(label)
            subset_i += n_features

    if pca:
        if os.path.exists(INDEX_FILENAME_PCA):
            mat = faiss.read_VectorTransform(INDEX_FILENAME_PCA)
        else:
            mat = faiss.PCAMatrix (FEATURES_NUMBER, PCA_FEATURES)

            print("PCA training... started")
            mat.train(train_subset)
            print("PCA training... finished")
            
            faiss.write_VectorTransform(mat, INDEX_FILENAME_PCA)

    if pca:
        print("PCA transformation... started")
        train_subset = mat.apply_py(train_subset) if pca else train_subset
        print("PCA transformation... finished")

    cpu_index = faiss.IndexFlatL2(PCA_FEATURES if pca else FEATURES_NUMBER) 
    #cpu_index =  faiss.index_factory(PCA_FEATURES if pca else FEATURES_NUMBER, "IVF4096,Flat")
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co) if gpu else cpu_index#, co)
    #nlist = 1000
    if train:
        print("Training index... started")
        #quantizer = faiss.IndexFlatL2(FEATURES_NUMBER)  # the other index
        #index = faiss.IndexIVFFlat(quantizer, FEATURES_NUMBER, nlist, faiss.METRIC_L2)
        # faster, uses more memory

        assert not index.is_trained
        index.train(train_subset)
        assert index.is_trained
        print("Training index... finished")

    subset_i = 0
    print("Adding {} train features to index".format(len(label_features)))
    for label, features in tqdm(label_features.items()):
        n_features = features.shape[0]
        f = features[:n_features]
        index.add(mat.apply_py(f) if pca else f)
        for n_feature in range(n_features):
            index_dict[subset_i+n_feature] = int(label)
        subset_i += n_features

    faiss.write_index(faiss.index_gpu_to_cpu(index) if gpu else index, INDEX_FILENAME)

    with open(INDEX_FILENAME_PK, 'wb') as fp:
        pickle.dump(index_dict, fp)

print("Indexed vectors {}".format(index.ntotal))
index.nprobe = 100

files = sorted(glob.glob(FEATURES_NPY))

if not args.extract_train_nn:
    test = np.empty((len(files), FEATURES_NUMBER), dtype=np.float32)
    subset_i = 0
    test_ids = []
    print("Loading test features for search")
    for file_name in tqdm(files):
        features = np.load(file_name)
        test_id = file_name.split('/')[-1].split('.')[0]
        if len(test_id) != 16:
            continue
        test_ids.append(test_id)
        test[subset_i] = features
        subset_i += 1
    index_dict[-1] = -1
    test = test[:subset_i]
    print("Search... started")  
    D, I = index.search(mat.apply_py(test) if pca else test, args.top_k)
    print("Search... finished")
    suffix = ""
else:
    label_features = { }
    n_train_set = 0
    for file_name in tqdm(files):
        label = file_name.split('/')[-1].split('.')[0]
        if len(label) == 16:
            continue
        features = np.load(file_name)
        assert features.shape[1] == FEATURES_NUMBER
        label_features[label] = features
        n_train_set += features.shape[0]
    train_set = np.empty((n_train_set, FEATURES_NUMBER), dtype=np.float32)
    print("Search... started")  
    test_ids = []
    D = np.empty((n_train_set, args.top_k+1), np.float32)
    I = np.empty((n_train_set, args.top_k+1), np.int32)
    i = 0
    for label, features in tqdm(label_features.items()):
        print(label)
        n_features = features.shape[0]
        _D, _I = index.search(mat.apply_py(features) if pca else features, args.top_k+1)
        D[i:i+n_features,...] = _D
        I[i:i+n_features,...] = _I
        i += n_features
        test_ids.extend([label] * n_features)
    print("Search... finished {} train items and {} items".format(i, n_train_set))  
    suffix = "_train"

landmarks = np.vectorize(lambda i: index_dict[i])(I)

os.makedirs(args.results_dir, exist_ok=True)

np.save(INDEX_FILENAME_PRE + ".distances" + suffix, D)
np.save(INDEX_FILENAME_PRE + ".landmarks" + suffix, landmarks)
with open(INDEX_FILENAME_PRE + ".testids" + suffix, 'wb') as fp:
    pickle.dump(test_ids, fp)

if args.print_results != 0:
    for i, test_id in enumerate(test_ids[:args.print_results]):
        print("{} -> {} {}".format(test_id, landmarks[i], D[i,0]))