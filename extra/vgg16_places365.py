from __future__ import absolute_import
import keras
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://s3-us-west-2.amazonaws.com/kaggleglm/vgg16_places365.h5'
WEIGHTS_PATH_NO_TOP = 'https://s3-us-west-2.amazonaws.com/kaggleglm/vgg16_places365_notop.h5'

def preprocess_input(x, mode='tf'):
    x[:, :, 0] -= 104.006
    x[:, :, 1] -= 116.669
    x[:, :, 2] -= 122.679
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    return x

def VGG16Places365(include_top=True,
                          weights='imagenet',
                          classes=365,
                          **kwargs):

    model = VGG16(
        weights=None, classes=classes, include_top=include_top, **kwargs)

    if weights:
        if include_top:
            weights_path = get_file(
                'vgg16_places365.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='7944d606893ead4c74336c354045d248')

            model.load_weights(weights_path)
        else:
            weights_path = get_file(
                'vgg16_places365_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='d5f7340fd4cb346f59500788ee97b087')

            model.load_weights(weights_path)

    return model
