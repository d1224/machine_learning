# coding: utf-8
# author: zbdai time:2020/8/3

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np


class PositionalEncodingLayer(Layer):

    def __init__(self, d_model, **kwargs):

        self.d_model = d_model

        super(PositionalEncodingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.position_info = self.add_weight(shape = (input_shape[1], self.d_model),
                                             initializer = 'zeros',
                                             trainable = False,
                                             dtype = K.floatx,
                                             name = "position_info")
        
        super(PositionalEncodingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        for i in range(inputs.shape[1]):
            for j in range(self.d_model):
                self.position_info[i, j] = i / np.power(10000, 2 * i / self.d_model)

        self.position_info[:, 0::2] = np.sin(self.position_info[:, 0::2])
        self.position_info[:, 1::2] = np.cos(self.position_info[:, 1::2])

        return self.position_info

    def compute_output_shape(self, input_shape):
        return input_shape



