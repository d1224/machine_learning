# coding: utf-8
# author: zbdai time:2020/8/3
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class EmbeddingLayer(Layer):

    def __init__(self, vocabulary_size, d_model, **kwargs):

        self.vocabulary_size = vocabulary_size
        self.d_model = d_model

        super(EmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(shape = (self.vocabulary_size, self.d_model),
                                                initializer = 'glorot_uniform',
                                                name = 'embedding_matrix')

        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        output = K.gather(self.embedding_matrix, inputs)

        output *= self.d_model ** 0.5

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape + (self.d_model, )

        return output_shape