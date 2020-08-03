# coding: utf-8
# author: zbdai time:2020/8/3

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

class ScaledDotProductAttention(Layer):

    def __init__(self, masking = True, future_masking = False, dropout = 0., **kwargs):
        self.masking = masking
        self.future_masking = future_masking
        self.dropout = dropout
        self.mask_num = -2 ** 32 + 1

        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)

        output = inputs + masks * self.mask_num
        return output

    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])

        tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()

        future_mask = tf.tile(tf.expand_dims) * self.mask_num

        padding = tf.ones_like(future_mask) * self.mask_num
        output = tf.where(tf.equal(self.future_masking, 0), padding, inputs)

        return output

    def call(self, inputs, **kwargs):
        if self.masking:
            q, k, v, masks = inputs
        else:
            q, k, v = inputs

        matmul = K.batch_dot(q, tf.transpose(k, [0, 2, 1]))
        scale = matmul / (int(q.shape[-1]) ** 0.5)

        if self.masking:
            scale = self.mask(scale, masks)

        if self.future_masking:
            scale = self.future_mask(scale)

        out_softmax = K.softmax(scale)

        out_dropout = K.dropout(out_softmax, self.dropout)
        output = K.batch_dot(out_dropout, v)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape