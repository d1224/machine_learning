# coding: utf-8
# author: zbdai time:2020/8/3

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

from transformer.layer.FeedForwardNetworkLayer import FeedForwardNetworkLayer
from transformer.layer.LayerNormalization import LayerNormalization
from transformer.layer.PositionalEncodingLayer import PositionalEncodingLayer
from transformer.layer.attention.SelfAttention import SelfAttention


class Transformer(Layer):

    def __init__(self, d_vocab, d_model, head_num = 8, encoder_stack = 6, decoder_stack = 6,
                 d_ff = 2048, dropout = 0.1, **kwargs):
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.head_num = head_num
        self.encoder_stack = encoder_stack
        self.decoder_stack = decoder_stack
        self.d_ff = d_ff
        self.dropout = dropout

        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(shape = (self.d_vocab, self.d_model),
                                                initializer = 'glorot_uniform',
                                                trainable = True,
                                                name = 'embedding_matrix')

        super(Transformer, self).build(input_shape)

    def encoder(self, inputs):

        masks = K.equal(inputs, 0)

        embedding = K.gather(self.embedding_matrix, inputs)
        embedding *= self.d_model ** 0.5
        position_info = PositionalEncodingLayer(self.d_model)(embedding)

        encoding = embedding + position_info
        encoding = K.dropout(encoding, self.dropout)

        for _ in range(self.encoder_stack):
            self_attention = SelfAttention(self.head_num, self.d_model // self.head_num)
            attention_input = [encoding, encoding, encoding, masks]

            attention_out = self_attention(attention_input)
            attention_out += encoding
            attention_out = LayerNormalization()(attention_out)

            ff_network = FeedForwardNetworkLayer(self.d_model, self.d_ff)
            ff_out = ff_network(attention_out)

            ff_out += attention_out
            encoding = LayerNormalization()(ff_out)

        return encoding, masks

    def decoder(self, inputs):
        de_inputs, en_encoding, en_masks = inputs

        de_masks = K.equal(de_inputs, 0)

        embedding = K.gather(self.embedding_matrix, de_inputs)
        embedding *= self.d_model ** 0.5

        position_info = PositionalEncodingLayer(self.d_model)(embedding)

        encoding = embedding + position_info
        encoding = K.dropout(encoding, self.dropout)

        for _ in range(self.decoder_stack):
            masked_attetnion = SelfAttention(self.head_num, self.d_model // self.head_num, future_masking = True)
            masked__attention_input = [encoding, encoding, encoding, de_masks]
            masked_attention_out = masked_attetnion(masked__attention_input)

            masked_attention_out += encoding
            masked_attention_out = LayerNormalization()(masked_attention_out)

            self_attention = SelfAttention(self.head_num, self.d_model // self.head_num)
            attention_input = [masked_attention_out, en_encoding, en_encoding, en_masks]
            attention_out = self_attention(attention_input)

            attention_out += masked_attention_out
            attention_out = LayerNormalization(attention_out)

            ff_network = FeedForwardNetworkLayer(self.d_model, self.d_ff)
            ff_out = ff_network(attention_out)

            ff_out += attention_out
            encoding = LayerNormalization()(ff_out)

        linear_projection = K.dot(encoding, K.transpose(self.embedding_matrix))
        output = K.softmax(linear_projection)

        return output

    def call(self, inputs, **kwargs):
        encoder_in, decoder_in = inputs
        encoder_encoding, encoder_masks = self.encoder(encoder_in)

        decoder_out = self.decoder([decoder_in, encoder_encoding, encoder_masks])

        return decoder_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.d_vocab)
