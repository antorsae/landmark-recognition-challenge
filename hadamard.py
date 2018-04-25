# Keras layer implementation of "Fix your classifier: the marginal value of training the last weight layer"
# https://arxiv.org/abs/1801.04540

from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras.initializers import Constant, RandomUniform
import numpy as np
from scipy.linalg import hadamard
import math

class HadamardClassifier(Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        super(HadamardClassifier, self).__init__(**kwargs)

    def build(self, input_shape):

        hadamard_size = 2 ** int(math.ceil(math.log(max(input_shape[1], self.output_dim), 2)))
        self.hadamard = K.constant(
            value=hadamard(hadamard_size, dtype=np.int8)[:input_shape[1], :self.output_dim])
        
        init_scale = 1. / math.sqrt(self.output_dim)

        self.scale = self.add_weight(name='scale', 
                                      shape=(1,),
                                      initializer=Constant(init_scale),
                                      trainable=True)

        self.bias  = self.add_weight(name='bias', 
                                      shape=(self.output_dim,),
                                      initializer=RandomUniform(-init_scale, init_scale),
                                      trainable=True)

        super(HadamardClassifier, self).build(input_shape)

    def call(self, x):
        output = K.l2_normalize(x, axis=-1)
        output = -self.scale * K.dot(output, self.hadamard)
        output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)