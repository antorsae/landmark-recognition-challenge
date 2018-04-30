# Keras layer implementation of "Fix your classifier: the marginal value of training the last weight layer"
# by Andres Torrubia, licensed under GPL 3: https://www.gnu.org/licenses/gpl-3.0.en.html
# https://arxiv.org/abs/1801.04540

from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras.initializers import Constant, RandomUniform
import numpy as np
from scipy.linalg import hadamard
import math

class HadamardClassifier(Layer):

    def __init__(self, output_dim, activation=None, use_bias=True, l2_normalize=True, output_raw_logits=False, **kwargs):
        self.output_dim        = output_dim
        self.activation        = activations.get(activation)
        self.use_bias          = use_bias
        self.l2_normalize      = l2_normalize
        self.output_raw_logits = output_raw_logits
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

        if self.use_bias:
            self.bias  = self.add_weight(name='bias', 
                                          shape=(self.output_dim,),
                                          initializer=RandomUniform(-init_scale, init_scale),
                                          trainable=True)

        super(HadamardClassifier, self).build(input_shape)

    def call(self, x, training=None):
        is_training = training not in {0, False}
        output = K.l2_normalize(x, axis=-1) if self.l2_normalize else x
        output = -self.scale * K.dot(output, self.hadamard) # pity .dot requires both tensors to be same type, the last one could be int8
        if self.output_raw_logits and not is_training:
            output_logits = -self.scale * K.dot(x, self.hadamard) # probably better to reuse output * l2norm
        if self.use_bias:
            output = K.bias_add(output, self.bias)
            if self.output_raw_logits and not is_training:
                output_logits = K.bias_add(output_logits, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        if self.output_raw_logits:
            return [output, output_logits if not is_training else output]
        return output

    def compute_output_shape(self, input_shape):
        if self.output_raw_logits:
            return [(input_shape[0], self.output_dim)] * 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'l2_normalize': self.l2_normalize,
            'output_raw_logits' : self.output_raw_logits
        }
        base_config = super(HadamardClassifier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))