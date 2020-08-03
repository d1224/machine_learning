# coding: utf-8
# author: zbdai time:2020/8/3

from tensorflow.keras.layers import Layer
import tensorflow as tf


class LayerNormalization(Layer):

    def __init__(self, epsilon = 1e-8, **kwargs):
        self.epsilon = epsilon

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(shape = (input_shape[-1], ),
                                    initializer = 'zeros',
                                    name = 'beta')

        self.gama = self.add_weight(shape = (input_shape[-1], ),
                                    initializer = 'one',
                                    name = 'gama')

        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):

        m, var = tf.nn.moments(inputs, [-1], keepdims = True)

        x = (inputs- m) / ((var + self.epsilon) ** 0.5)
        output = self.gama * x + self.beta

        return output

    def compute_output_shape(self, input_shape):
        return input_shape
