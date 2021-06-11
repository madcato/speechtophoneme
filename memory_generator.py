#!/usr/bin/env python

# From: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import keras
from utils import text_to_int_sequence
from char_map import get_number_of_char_classes
import sys
from keras.utils import *

num_classes = get_number_of_char_classes()

# THIS CLASS DOEN'T WORK TO FINE. IT'S SLOWER THAN NORMAL VERSION AND CONSUME MORE MEMORY. !!??

class PhonemeMemoryDataGenerator(Sequence):
    ''' This class is a generator that stores all the data in memory 
        instead in disk.
    '''
    def __init__(self, dataframe, batch_size=32, empty_label=num_classes):
        'Initialization'
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.indexes = self.dataframe.index
        self.empty_label = int(empty_label)
        self.on_epoch_end()
        self.init_data()

    def init_data(self):
        '''Initialize all the data loading it to memory
        '''
        wav_filenames = self.dataframe['wav_filename']
        texts = self.dataframe['transcript']
        self.__data_generation(wav_filenames, texts)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return

    def load(self, filename):
        return np.load(filename)

    def __data_generation(self, wav_filenames, texts):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # calculate necessary sizes
        data_length = len(wav_filenames)
        features = [self.load(wav_filenames[i]) for i in range(0, data_length)]
        phonemes = [self.load(texts[i]) for i in range(0, data_length)]
        max_length = max([features[i].shape[0] for i in range(0, data_length)])
        max_string_length = max([len(phonemes[i]) for i in range(0, data_length)])
        feat_dim = features[0].shape[1]  # should be mfcc coef
        # Initialization
        X_data = np.zeros([data_length, max_length, feat_dim])
        labels = np.ones([data_length, max_string_length]) * self.empty_label
        input_length = np.zeros([data_length, 1])
        label_length = np.zeros([data_length, 1])
        # Generate data
        for i, ID in enumerate(features):
            # calculate X & input_length
            feat = features[i]
            input_length[i] = feat.shape[0]
            X_data[i, :feat.shape[0], :] = feat
            # calculate labels & label_length
            label = np.array(text_to_int_sequence(phonemes[i]))
            labels[i, :len(label)] = label
            label_length[i] = len(label)
        self.X_data = X_data 
        self.labels = labels
        self.input_length = input_length
        self.label_length = label_length

    def __len__(self):
        'Denotes the number of batches per epoch' # return: samples / batch_len
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_data = self.X_data[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        input_length = self.input_length[index*self.batch_size:(index+1)*self.batch_size]
        label_length = self.label_length[index*self.batch_size:(index+1)*self.batch_size]
        outputs = {'ctc': np.zeros([self.batch_size])}
        inputs = {'the_input': X_data, 
                  'the_labels': labels, 
                  'input_length': input_length, 
                  'label_length': label_length
                 }
        return inputs, outputs
