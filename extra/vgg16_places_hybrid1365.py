from __future__ import absolute_import
import keras
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://s3-us-west-2.amazonaws.com/kaggleglm/vgg16_hybrid1365.h5'
WEIGHTS_PATH_NO_TOP = 'https://s3-us-west-2.amazonaws.com/kaggleglm/vgg16_hybrid1365_notop.h5'

def preprocess_input(x, mode='tf'):
    x[:, :, 0] -= 104.006
    x[:, :, 1] -= 116.669
    x[:, :, 2] -= 122.679
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    return x

def VGG16PlacesHybrid1365(include_top=True,
                          weights='imagenet',
                          classes=1365,
                          input_shape=(128,128,3),
                          **kwargs):

    model = VGG16(
        weights=None, classes=classes, input_shape=input_shape, include_top=include_top, **kwargs)

    if weights:
        if include_top:
            weights_path = get_file(
                'vgg16_hybrid1365.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='3ddd2396e124c93143b9bd5d1835e10e')

            model.load_weights(weights_path)
        else:
            weights_path = get_file(
                'vgg16_hybrid1365_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='696badfd31f1195212e3501c8edfc4e4')

            model.load_weights(weights_path)

    return model
