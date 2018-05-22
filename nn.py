import glob
import numpy as np
import faiss                     # make faiss available
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import math
import os
import pickle

FEATURES_NUMBER = 16384
PCA_FEATURES    = 512

train = False

INDEX_FILENAME     = "features_AXception-cs256.index" 
INDEX_FILENAME_PK  = INDEX_FILENAME + '.pk'
INDEX_FILENAME_PCA = INDEX_FILENAME + '.pca'

res = faiss.StandardGpuResources()  # use a single GPU
#co = faiss.GpuClonerOptions()
# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
#co.useFloat16 = True

if os.path.exists(INDEX_FILENAME):
    index = faiss.read_index(INDEX_FILENAME)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)#, co)

    mat = faiss.read_VectorTransform(INDEX_FILENAME_PCA) # todo calculate it if not there

    with open(INDEX_FILENAME_PK, 'rb') as fp:
        index_dict = pickle.load(fp)
else:
    files = glob.glob("features/AXception-cs256/*.npy")
    index_dict = { }
    label_features = { }
    i = 0
    n_train_subset = 0
    for file_name in tqdm(files):
        label = file_name.split('/')[-1].split('.')[0]
        if len(label) == 16:
            continue
        features = np.load(file_name)
        label_features[label] = features
        n_train_subset += max(1, features.shape[0] // 5)

    print(n_train_subset)

    train_subset = np.empty((n_train_subset, FEATURES_NUMBER), dtype=np.float32)

    subset_i = 0
    for label, features in label_features.items():
        n_features = max(1, features.shape[0] // 5)
        train_subset[subset_i:subset_i+n_features] = features[:n_features]
        for n_feature in range(n_features):
            index_dict[subset_i+n_feature] = int(label)
        subset_i += n_features

    if os.path.exists(INDEX_FILENAME_PCA):
        mat = faiss.read_VectorTransform(INDEX_FILENAME_PCA)
    else:
        mat = faiss.PCAMatrix (FEATURES_NUMBER, PCA_FEATURES)

        print("PCA training... started")
        mat.train(train_subset)
        print("PCA training... finished")
    
        faiss.write_VectorTransform(mat, INDEX_FILENAME_PCA)

    if train:
        print("PCA transformation... started")
        train_subset = mat.apply_py(train_subset)
        print("PCA transformation... finished")

    index = faiss.IndexFlatL2(PCA_FEATURES) # faiss.index_factory(PCA_FEATURES, "IVF4096,Flat")
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)#, co)
    #nlist = 1000
    if train:
        print("Training index... started")
        #quantizer = faiss.IndexFlatL2(FEATURES_NUMBER)  # the other index
        #index = faiss.IndexIVFFlat(quantizer, FEATURES_NUMBER, nlist, faiss.METRIC_L2)
        # faster, uses more memory


        assert not gpu_index.is_trained
        gpu_index.train(train_subset)
        assert gpu_index.is_trained
        print("Training index... finished")

    for label, features in tqdm(label_features.items()):
        n_features_s = max(1, features.shape[0] // 5) if train else 0
        n_features_e = features.shape[0] - n_features_s
        gpu_index.add(mat.apply_py(features[n_features_s:n_features_s+n_features_e]))
        for n_feature in range(n_features_e):
            index_dict[subset_i+n_feature] = int(label)
        subset_i += n_features_e

    if False:
        i = 0
        groups = 10
        index_groups = np.array_split(files_index, groups)
        index_subset = np.empty((math.ceil(len(files_index)/groups), FEATURES_NUMBER), dtype=np.float32)

        for files_group in tqdm(index_groups):
            subset_i = 0
            for file_name in tqdm(files_group, leave=False):
                features = np.load(file_name)
                train_id = file_name.split('/')[-1].split('.')[0]
                if len(train_id) != 16:
                    continue
                index_subset[subset_i] = features
                subset_i += 1   
                index_dict[i] = train_id
                i += 1 
            gpu_index.add(index_subset[:subset_i])

    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), INDEX_FILENAME)

    with open(INDEX_FILENAME_PK, 'wb') as fp:
        pickle.dump(index_dict, fp)

files = glob.glob("features/AXception-cs256/*.npy")[:32]
test = np.empty((len(files), FEATURES_NUMBER), dtype=np.float32)
subset_i = 0
test_ids = []
for file_name in tqdm(files):
    features = np.load(file_name)
    test_id = file_name.split('/')[-1].split('.')[0]
    if len(test_id) != 16:
        continue
    test_ids.append(test_id)
    test[subset_i] = features
    subset_i += 1
index_dict[-1] = -1
D, I = index.search(mat.apply_py(test), 10)
landmarks = np.vectorize(lambda i: index_dict[i])(I)

#print(D)
for i, test_id in enumerate(test_ids):
    print("{} -> {}".format(test_id, landmarks[i]))