import glob
import numpy as np
from annoy import AnnoyIndex
from keras.utils.data_utils import get_file

FEATURES_NUMBER = 16384
ANNOY_INDEX = AnnoyIndex(FEATURES_NUMBER)

lfh = open("labels.ann", "w")

INDOOR_IMAGES_URL = 'https://s3-us-west-2.amazonaws.com/kaggleglm/train_indoor.txt'
INDOOR_IMAGES_PATH = get_file(
    'train_indoor.txt',
    INDOOR_IMAGES_URL,
    cache_subdir='models',
    file_hash='a0ddcbc7d0467ff48bf38000db97368e')
indoor_images = set(open(INDOOR_IMAGES_PATH, 'r').read().splitlines())

indoors = set(open("train_indoor.txt").read().splitlines())
files = glob.glob("features/AXception-cs192/*.npy")
i = 0
for file_name in files:
    vectors = np.load(file_name)
    train_id = file_name.split('/')[-1].split('.')[0]
    if len(train_id) == 16:
        continue
    if train_id in indoor_images:
        continue

    label = int(file_name.split('/')[-1].split('.')[0])
    for j in range(len(vectors)):
        lfh.write("%s %s\n" % (i, label))
        ANNOY_INDEX.add_item(i, vectors[j][:FEATURES_NUMBER])
        i = i + 1

ANNOY_INDEX.build(200)
ANNOY_INDEX.save('inxed.ann')
lfh.close()
