# coding: utf-8
# author: zbdai time:2020/8/3
import unittest
import tensorflow as tf


from tensorflow.keras.preprocessing import sequence

from keras.datasets import imdb
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import GlobalAveragePooling1D, Dropout, Dense, Embedding
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from transformer.layer.PositionalEncodingLayer import PositionalEncodingLayer
from transformer.layer.attention.SelfAttention import SelfAttention


class SelfAttentionTest(unittest.TestCase):

    def __init__(self):

        self.d_vocab = 5000
        self.max_len = 256
        self.d_model = 512
        self.batch_size = 128
        self.epochs = 10

    def getTrainData(self):
        print('data downloading and pre-processing ...')

        (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen = self.max_len, num_words = self.d_vocab)

        x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def buildModel(self):
        print('model building ... ')
        inputs = Input(shape=(self.max_len, ), name='inputs')
        masks = Input(shape=(self.max_len, ), name='masks')

        embedding = Embedding(self.d_vocab, self.d_model)(inputs)

        encoding = PositionalEncodingLayer(self.d_model)(embedding)
        encoding += embedding

        x = SelfAttention(8, 64)([encoding, encoding, encoding, masks])
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation = 'relu')(x)

        out = Dense(2, activation='softmax')(x)

        model = Model(inputs=[inputs, masks], outputs=out)
        model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def trainModel(self):
        print('model training ...')
        x_train, y_train, x_test, y_test = self.getTrainData()
        model = self.buildModel()

        es = EarlyStopping(patience=5)

        x_train_masks = tf.equal(x_train, 0)
        model.fit([x_train, x_train_masks], y_train, batch_size=self.batch_size,
                  epochs=self.epochs, validation_split=0.2,
                  callbacks=[es])

        x_test_masks = tf.equal(x_train, 0)
        test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=self.batch_size,
                                      verbose=0)

        print("loss on Test: %.4f" % test_metrics[0])
        print("accu on Test: %.4f" % test_metrics[1])

    def test_train(self):
        print('train start ... ')
        self.trainModel()

        print('train end ... ')
