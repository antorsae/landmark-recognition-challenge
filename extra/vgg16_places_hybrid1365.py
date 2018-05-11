import keras
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://s3-us-west-2.amazonaws.com/kaggleglm/vgg16_hybrid1365.h5'
WEIGHTS_PATH_NO_TOP = 'https://s3-us-west-2.amazonaws.com/kaggleglm/vgg16_hybrid1365_notop.h5'


def VGG16PlacesHybrid1365(include_top=True,
                          weights='imagenet',
                          classes=1365,
                          **kwargs):

    model = VGG16(weights=None, classes=classes, **kwargs)

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
