# coding: utf-8
# author: zbdai time:2020/8/3

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class FeedForwardNetworkLayer(Layer):

    def __init__(self, d_model, d_ff, trainable = True, **kwargs):
        self.d_model = d_model
        self.d_ff = d_ff
        self.trainable = trainable

        super(FeedForwardNetworkLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(shape = (input_shape[-1], self.d_ff),
                                  initializer = 'glorot_uniform',
                                  trainable = self.trainable,
                                  name = 'weight1')
        self.w2 = self.add_weight(shape = (self.d_ff, self.d_model),
                                  input_shape = 'glorot_uniform',
                                  trainable = self.trainable,
                                  name = 'weigh2')

        self.b1 = self.add_weight(shape=(self.d_ff, ),
                                  initializer='uniform',
                                  trainable=self.trainable,
                                  name='bais1')
        self.b2 = self.add_weight(shape=(self.d_model, ),
                                  input_shape='uniform',
                                  trainable=self.trainable,
                                  name='bais2')

        super(FeedForwardNetworkLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        f1 = K.relu(K.dot(inputs, self.w1) + self.b1)

        output = K.dot(f1, self.w2) + self.b2

        return output

    def compute_output_shape(self, input_shape):
        return self.d_model