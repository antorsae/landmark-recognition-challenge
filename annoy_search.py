import glob
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm

FEATURES_NUMBER = 16384
ANNOY_INDEX = AnnoyIndex(FEATURES_NUMBER)
ANNOY_INDEX.load('inxed.ann')

indoors = set(open("test_indoor.txt").read().splitlines())
ofh = open("annoy_res.txt", "w")

MAP = dict()
labels = open('labels.ann').read().splitlines()
for label in labels:
    idx, label = label.split(' ')
    MAP[idx] = label

test_files = glob.glob('features/AXception-cs192/*.npy')

for test_file in tqdm(test_files):
    test_id = test_file.split('/')[-1].split('.')[0]
    if test_id in indoors:
        continue
    if len(test_id) != 16:
        continue

    v = np.load(test_file)[:FEATURES_NUMBER]
    result = ANNOY_INDEX.get_nns_by_vector(
        v, 1, search_k=-1, include_distances=True)

    ofh.write("%s %s %f\n" % (test_id, MAP[str(result[0][0])], result[1][0]))

ogh.close()
