# coding: utf-8
# author: zbdai time:2020/8/3

import unittest

from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from transformer.callback.TransformerCallback import TransformerCallback
from transformer.utils.Utils import Utils


class TransformerTest(unittest.TextCase):

    def __init__(self):
        self.dialogs_path = 'xiaohuangji50w_fenciA.conv'
        self.dialogs_size = 20000
        self.vocab_size = 3000
        self.max_seq_len = 10
        self.d_model = 512
        self.batch_size = 256
        self.epochs = 10

    def build_model(self):
        print('model building ... ')
        encoder_inputs = Input(shape=(self.max_seq_len,), name='encoder_inputs')
        decoder_inputs = Input(shape=(self.max_seq_len,), name='decoder_inputs')

        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_inputs)

        model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                      loss='categorical_crossentropy')

        return model

    def train_model(self):
        print('model training ... ')

        callback = TransformerCallback(self.d_model)

        question_input, answer_input, decoder_target = Utils.load_data(self.dialogs_path,
                                                                       self.dialogs_size,
                                                                       self.vocab_size,
                                                                       max_len=self.max_seq_len)

        model = self.build_model()
        model.fit([question_input, answer_input], decoder_target, batch_size=self.batch_size,
                  epochs=self.epochs, validation_split=2, callbacks=[callback])


