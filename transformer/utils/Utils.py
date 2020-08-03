# coding: utf-8
# author: zbdai time:2020/8/3
import re

from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np


class Utils:

    @staticmethod
    def load_data(data_path, dialog_size, vocab_size=5000, max_len=10):
        print('data loading and tokenizing ... ')

        with open(data_path, 'r') as f:
            data = f.read()

            relu = re.comple("E\nM (.*?)\nM (.*?)\n")

            match_dialog = re.findall(relu, data)

            dialog_size = len(match_dialog) - 1 if dialog_size > len(match_dialog) else dialog_size
            dialogs = match_dialog[:dialog_size]

        questions = [dia[0] for dia in dialogs]
        answers = ['<start>/' + dia[1] + '/<stop>' for dia in dialogs]

        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(questions + answers)

        questions_seqs = tokenizer.texts_to_sequences(questions)
        questions_seqs = pad_sequences(questions_seqs, max_len=max_len)

        answers_seqs = tokenizer.texts_to_sequences((answers))
        answers_seqs = pad_sequences(answers_seqs, max_len=max_len)

        decoder_target = np.zeros((len(answers_seqs), max_len, vocab_size), dtype='float32')

        for i, seq in enumerate(answers_seqs):
            for j, index in enumerate(seq):
                if j > 0: decoder_target[i, j-1, index-1] = 1
                if index == 0: break

        return questions_seqs, answers_seqs, decoder_target

    def label_smoothing(inputs, epsilon=0.1):
        output_dim = inputs.shape[-1]
        smooth_label = (1 - epsilon) * inputs + (epsilon / output_dim)
        return smooth_label
