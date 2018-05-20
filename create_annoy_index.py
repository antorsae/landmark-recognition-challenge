import glob
import numpy as np
from annoy import AnnoyIndex

FEATURES_NUMBER = 16384
ANNOY_INDEX = AnnoyIndex(FEATURES_NUMBER)

lfh = open("labels.ann", "w")

files = glob.glob("features/AXception-cs192/*.npy")
i = 0
for file_name in files:
    vectors = np.load(file_name)
    if len(file_name.split('/')[-1].split('.')[0]) == 16:
        continue
    label = int(file_name.split('/')[-1].split('.')[0])
    for j in range(len(vectors)):
        lfh.write("%s %s\n" % (i, label))
        ANNOY_INDEX.add_item(i, vectors[j][:FEATURES_NUMBER])
        i = i + 1

ANNOY_INDEX.build(200)
ANNOY_INDEX.save('inxed.ann')
lfh.close()
