# coding: utf-8
# author: zbdai time:2020/8/3

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

class TransformerCallback(Callback):

    def __init__(self, d_model, step_num=0, warmup_steps=4000, verbose=False, **kwargs):
        self.d_model = d_model
        self.step_num = step_num
        self.warmup_steps = warmup_steps
        self.verbose = verbose

        super(TransformerCallback, self).__init__(**kwargs)

    def on_train_batch_begin(self, batch, logs=None):
        logs = logs or {}
        init_lr = self.d_model ** -.5 * self.warmup_steps ** -1.5

        K.set_value(self.model.optimizer.lr, init_lr)

    def on_predict_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.step_num += 1
        lrate = self.d_model ** -.5 * K.minimum(self.step_num ** -.5, self.step_num * self.warmup_steps ** -1.5)

        K.set_value(self.model.optimizer.lr, lrate)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            lrate = K.get_value(self.model.optimizer.lr)
            print(f"epoch: {epoch}, lr: {lrate}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        logs['lr'] = K.get_value(self.model.optimizer.lr)
