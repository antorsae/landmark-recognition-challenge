import glob
import sys
import csv
import jpeg4py as jpeg
from PIL import Image
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.utils.data_utils import get_file
import cv2

# USAGE:
# CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=tensorflow python indoor_outdoor_detector.py 0
# CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=tensorflow python indoor_outdoor_detector.py 1
# ...
# CUDA_VISIBLE_DEVICES=$TOTAL_GPUS KERAS_BACKEND=tensorflow python indoor_outdoor_detector.py $TOTAL_GPUS
# Setup TOTAL_GPUS below

def preprocess_input(x, mode='tf'):
    if mode == 'th':
        x[:, 0, :, :] -= 104.006
        x[:, 1, :, :] -= 116.669
        x[:, 2, :, :] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, 0] -= 104.006
        x[:, :, 1] -= 116.669
        x[:, :, 2] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
    return x

INDOOR_CLASSES = [
    1, 2, 3, 6, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28,
    29, 31, 34, 35, 37, 38, 39, 43, 44, 45, 46, 50, 51, 52, 54, 55, 56, 60, 61,
    63, 64, 65, 69, 70, 72, 75, 80, 82, 85, 88, 89, 90, 92, 93, 95, 96, 98, 99,
    100, 101, 102, 106, 114, 115, 120, 121, 122, 124, 126, 128, 129, 130, 131,
    133, 134, 135, 137, 139, 146, 147, 148, 155, 156, 160, 162, 165, 168, 169,
    172, 176, 177, 179, 182, 185, 188, 195, 196, 198, 202, 203, 208, 210, 211,
    212, 215, 217, 219, 222, 225, 228, 235, 236, 238, 239, 240, 241, 244, 246,
    248, 250, 253, 255, 261, 262, 264, 267, 269, 274, 280, 281, 282, 284, 285,
    295, 297, 298, 300, 302, 303, 311, 315, 317, 318, 320, 321, 322, 324, 325,
    328, 329, 331, 332, 335, 336, 337, 343, 346, 352, 358, 363
]
PB_FILE_URL = 'https://s3-us-west-2.amazonaws.com/kaggleglm/vgg16_places365_with_top.pb'
PB_FILE_PATH = get_file(
    'vgg16_places365_with_top.pb',
    PB_FILE_URL,
    cache_subdir='models',
    file_hash='4c40dd00cb94a5a92ae7eaf23292527b')
#PB_FILE_PATH = 'bin/vgg16_places365_with_top2.pb'
CROP_SIZE = 224
BATCH_SIZE = 32
GPU = int(sys.argv[1])
TOTAL_GPUS = 8
FILES = glob.glob("test-dl/*.jpg")

out_fh = open('csv/test_indoor_outdoor.csv.%s' % GPU, 'wb')
writer = csv.writer(out_fh)

images = tf.placeholder(
    "float", [None, CROP_SIZE, CROP_SIZE, 3], name="images")
with open(PB_FILE_PATH, mode='rb') as in_fh:
    file_content = in_fh.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(file_content)

tf.import_graph_def(graph_def, input_map={"input_1": images})

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess = tf.Session()


def process_item(item):

    load_img_fast_jpg = lambda img_path: jpeg.JPEG(img_path).decode()
    load_img = lambda img_path: np.array(Image.open(img_path))

    def try_load_PIL(item):
        try:
            img = load_img(item)
            return img
        except Exception:
            print('Decoding error:', item)
            return None

    loaded_pil = loaded_fast_jpg = False
    try:
        img = load_img_fast_jpg(item)
        loaded_fast_jpg = True
    except Exception:
        img = try_load_PIL(item)
        if img is None: return None
        loaded_pil = True

    shape = list(img.shape[:2])

    # some images may not be downloaded correctly and are B/W, discard those
    if img.ndim != 3:
        #if args.verbose:
        #    print('Ndims !=3 error:', item)
        if not loaded_pil:
            img = try_load_PIL(item)
            if img is None: return None, None, item
            loaded_pil = True
        if img.ndim == 2:
            img = np.stack((img, ) * 3, -1)
        if img.ndim != 3:
            return None

    if img.shape[2] != 3:
        #if args.verbose:
        #    print('More than 3 channels error:', item)
        if not loaded_pil:
            img = try_load_PIL(item)
            if img is None: return None, None, item
            loaded_pil = True
        return None

    img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))

    #img = imagenet_utils.preprocess_input(img.astype(np.float32), mode='tf')
    img = preprocess_input(img.astype(np.float32))
    return img


batch = []
idx = []

for file_path in tqdm(FILES[GPU::TOTAL_GPUS]):
    image_to_test = process_item(file_path)
    if image_to_test is not None:
        batch.append(image_to_test)
        idx.append(file_path.split('/')[-1].split('.')[0])

    if len(batch) == BATCH_SIZE:
        output = sess.run(
            sess.graph.get_tensor_by_name("import/predictions/Softmax:0"),
            feed_dict={
                images: batch,
            })

        for _, (id, item) in enumerate(zip(idx, output)):
            top5 = item.argsort()[-5:]
            votes_in = 0
            votes_out = 0
            for i in range(5):
                if top5[i] in INDOOR_CLASSES:
                    votes_in += 1  #item[i]
                else:
                    votes_out += 1  #item[i]
            row = [str(id)] + [int(votes_in == 5)] + [votes_in]
            writer.writerow(row)
        batch = []
        idx = []

if len(batch) > 0:
    output = sess.run(
        sess.graph.get_tensor_by_name("import/predictions/Softmax:0"),
        feed_dict={
            images: batch
        })

    for _, (id, item) in enumerate(zip(idx, output)):
        top5 = item.argsort()[-5:]
        votes_in = 0
        votes_out = 0
        for i in range(5):
            if top5[i] in INDOOR_CLASSES:
                votes_in += 1 #item[i]
            else:
                votes_out += 1 #item[i]

        row = [str(id)] + [int(votes_in == 5)] + [votes_in]
        writer.writerow(row)

out_fh.close()
