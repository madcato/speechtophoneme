#!/usr/bin/env python

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.initializers import random_normal
from keras.utils.conv_utils import conv_output_length
from keras.layers import GaussianNoise
import numpy as np
from char_map import get_number_of_char_classes

num_classes = get_number_of_char_classes()

def selu(x):
    # from Keras 2.0.6 - does not exist in 2.0.4
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
       x: A tensor or variable to compute the activation function for.
    # References
       - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)

def clipped_relu(x):
    return K.relu(x, max_value=20)

def create_model(input_dim=26, fc_size=512, dropout=[0.1, 0.1], output_dim=num_classes):
    """ Own model BN+SELU-FC+GRU+BN+DR

    Architecture:
        Batch Normalisation layer on the input data
        1 Fully connected layer of fc_size with SELU
        2 Fully connected layer of fc_size with Clipped Relu
        3 Dropout layers applied between the FC layers
        Batch Normalisation layer on the final FC output
        1 BiDirectional GRU layer with Clipped Relu
        1 Fully connected layer of fc_size with SELU
        1 Dropout layer
        1 Softmax out


    """
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})
    K.set_learning_phase(1)
    # Creates a tensor there are usually 26 MFCC
    input_data = Input(name='the_input', shape=(input_dim,))  # >>(?, 26)
    x = BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3,center=True,scale=True)(input_data)
    # First 3 FC layers
    init = random_normal(stddev=0.046875)
    x = Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=selu)(x)  # >>(?, 778, 2048)
    x = Dropout(dropout[0])(x)
    x = Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu)(x)  # >>(?, 778, 2048)
    x = Dropout(dropout[0])(x)
    x = Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu)(x)  # >>(?, 778, 2048)
    x = Dropout(dropout[0])(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)
    # Layer 4+5 Dense Layer & Softmax
    x = Dense(fc_size, activation=selu, kernel_initializer=init, bias_initializer=init)(x)
    x = Dropout(dropout[1])(x)
    y_pred = Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax")(x)
    model = Model(input=input_data, output=y_pred)
    print(model.summary())
    return model
