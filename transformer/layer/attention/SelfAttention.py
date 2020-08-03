# coding: utf-8
# author: zbdai time:2020/8/3

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

from transformer.layer.attention.ScaledDotProductAttention import ScaledDotProductAttention


class SelfAttention(Layer):

    def __init__(self, head_num, head_dim, dropout = .1, masking = True,
                 future_masking = False, trainable = True, **kwargs):
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.masking = masking
        self.future_masking = future_masking
        self.trainable = trainable

        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wq = self.add_weight(shape=(input_shape[0][-1], self.head_num * self.head_dim),
                                  initializer = 'glorot_uniform',
                                  trainable = self._trainalbe,
                                  name = 'weight_queries'
        )
        self.wk = self.add_weight(shape = (input_shape[1][-1], self.head_num * self.head_dim),
                                  initializer = 'glorot_uniform',
                                  trainable = self._trainalbe,
                                  name = 'weight_keys'
        )
        self.wv = self.add_weight(shape = (input_shape[2][-1], self.head_num * self.head_dim),
                                  initializer ='glorot_uniform',
                                  trainable =self._trainalbe,
                                  name = 'weight_values'
        )

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.masking:
            q, k, v, mask = input
        else:
            q, k, v = inputs

        q_linear = K.dot(q, self.wq)
        k_linear = K.dot(k, self.wk)
        v_linear = K.dot(v, self.wv)

        q_multi_head = tf.concat(tf.split(q_linear, self._n_head, axis=2), axis=0)
        k_multi_heads = tf.concat(tf.split(k_linear, self._n_head, axis=2), axis=0)
        v_multi_heads = tf.concat(tf.split(v_linear, self._n_head, axis=2), axis=0)

        layer_input = [q_multi_head, k_multi_heads, v_multi_heads]

        if self.masking:
            layer_input.append(mask)

        dot_product_attention = ScaledDotProductAttention(masking = self.masking,
                                                          future_masking = self.future_masking,
                                                          dropout = self.dropout)

        output = dot_product_attention(layer_input)
        output = tf.concat(tf.split(output, self.head_num, axis=0), axis=2)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

