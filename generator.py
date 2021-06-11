#!/usr/bin/env python

# From: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import keras
from utils import text_to_int_sequence, phoneme_to_int
from char_map import get_number_of_char_classes
import sys
from keras.utils import *

num_classes = get_number_of_char_classes()

class PhonemeDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, empty_label=num_classes, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.indexes = self.dataframe.index
        self.shuffle = shuffle
        self.empty_label = int(empty_label)
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load(self, filename):
        return np.load(filename)

    def __data_generation(self, wav_filenames, texts):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # calculate necessary sizes
        features = [self.load(wav_filenames[i]) for i in range(0, self.batch_size)]
        phonemes = [texts[i] for i in range(0, self.batch_size)]
        feat_dim = features[0].shape[0]  # should be mfcc coef
        # print("feat_dim: {}",format(feat_dim))
        # Initialization
        X_data = np.zeros([self.batch_size, feat_dim])
        labels = np.ones([self.batch_size, num_classes]) * self.empty_label
        # Generate data
        for i, ID in enumerate(features):
            # calculate X & input_length
            feat = features[i]
            X_data[i] = feat
            # calculate labels & label_length
            int_pho = phoneme_to_int(phonemes[i])
            labels[i] = keras.utils.to_categorical(int_pho, num_classes=num_classes, dtype='float32')
        # return the arrays
        outputs = labels
        inputs = X_data
        # print(inputs)
        # print(outputs)
        # print(inputs.shape)
        # print(outputs.shape)
        return (inputs, outputs)

    def __len__(self):
        'Denotes the number of batches per epoch' # return: samples / batch_len
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        wav_filenames = [self.dataframe['wav_filename'].iloc[k] for k in indexes]
        texts = [self.dataframe['transcript'].iloc[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(wav_filenames, texts)
        # print("step {}".format(index))
        # print("X.shape: {}".format(X.shape))
        # print("y.shape: {}".format(y.shape))
        return X, y
