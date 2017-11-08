import random
import os

import pandas as pd
import numpy as np
from PIL import Image

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import tensorflow as tf

from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, concatenate, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K

from position import POS_NUM
from img_utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, process_image

LEARNING_RATE = 1e-4
saved_h5_name = 'position_model.h5'
saved_json_name = 'position_model.json'


def build_model(learn_rate=LEARNING_RATE):
    print("Now we build the model")
    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    values = inputs

    conv_layer_1 = Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu')
    values = conv_layer_1(values)
    values = BatchNormalization()(values)

    conv_layer_2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')
    values = conv_layer_2(values)
    values = BatchNormalization()(values)

    conv_layer_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    values = conv_layer_3(values)
    values = BatchNormalization()(values)

    max_poll_layer = MaxPooling2D(pool_size=(2, 2))
    values = max_poll_layer(values)

    values = Flatten()(values)
    values = BatchNormalization()(values)

    hidden_layer = Dense(64, activation='relu')
    values = hidden_layer(values)

    pos_output_layer = Dense(POS_NUM, activation='relu')
    outputs = pos_output_layer(values)

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=adam)

    print("We finish building the model")
    model.summary()
    return model


train_list = [
    "/Volumes/CPSC587DATA/RecordedImg20171102233250",
    "/Volumes/CPSC587DATA/RecordedImg20171102234320",
    "/Volumes/CPSC587DATA/RecordedImg20171102235004",
    "/Volumes/CPSC587DATA/RecordedImg20171103000421",
    "/Volumes/CPSC587DATA/RecordedImg20171103004246",
]

test_list = [
    "/Volumes/CPSC587DATA/RecordedImg20171103010334",
]

