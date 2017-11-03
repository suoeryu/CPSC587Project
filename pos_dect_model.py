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

from car_position import POS_NUM
from img_utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, process_image

LEARNING_RATE = 1e-4


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

    hidden_layer = Dense(128, activation='relu')
    values = hidden_layer(values)

    pos_output_layer = Dense(POS_NUM, activation='relu')
    outputs = pos_output_layer(values)

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam)

    print("We finish building the model")
    # model.summary()
    return model


def train(model, img_folder, n_epoch=5):
    print("Training on", img_folder)
    csv_path = os.path.join(img_folder, 'info.csv')
    info = pd.read_csv(csv_path)

    img_list = []
    for _, row in info.iterrows():
        img = Image.open(os.path.join(img_folder, row['filename']))
        img = process_image(img)
        img_list.append(np.expand_dims(img, 0))
    inputs = np.concatenate(tuple(img_list), axis=0)
    car_pos_idx = np.asarray(info['car_pos_idx'])
    outputs = to_categorical(car_pos_idx, num_classes=POS_NUM)
    for epoch in range(n_epoch):
        loss = model.train_on_batch(inputs, outputs)
        print(loss)


def test(model, img_folder):
    csv_path = os.path.join(img_folder, 'info.csv')
    info = pd.read_csv(csv_path)

    img_list = []
    for _, row in info.iterrows():
        img = Image.open(os.path.join(img_folder, row['filename']))
        img = process_image(img)
        img_list.append(np.expand_dims(img, 0))
    inputs = np.concatenate(tuple(img_list), axis=0)
    outputs = model.predict(inputs)
    car_pos_idx = np.argmax(outputs, axis=1)
    y_true = np.asarray(info['car_pos_idx'])
    y_pred = car_pos_idx
    conf_mx = confusion_matrix(y_true, y_pred)
    print(conf_mx)
    print('Precision: ', precision_score(y_true, y_pred, average=None))
    print('Recall:    ', recall_score(y_true, y_pred, average=None))
    print('F1_score:  ', f1_score(y_true, y_pred, average=None))


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    m = build_model()
    train_img_folder_list = [
        '/Volumes/CPSC587DATA/TRAINING_IMAGES',
        "/Volumes/CPSC587DATA/RecordedImg20171102152944trained",
        "/Volumes/CPSC587DATA/RecordedImg20171102154113trained",
        "/Volumes/CPSC587DATA/RecordedImg20171102165209trained",
        "/Volumes/CPSC587DATA/RecordedImg20171102172503trained",
        "/Volumes/CPSC587DATA/RecordedImg20171102172849trained",
        "/Volumes/CPSC587DATA/RecordedImg20171102173621trained",
        "/Volumes/CPSC587DATA/RecordedImg20171102174708trained",
        "/Volumes/CPSC587DATA/RecordedImg2017_11_02_13_43_40trained",
        "/Volumes/CPSC587DATA/RecordedImg2017_11_02_14_03_04trained",
        "/Volumes/CPSC587DATA/RecordedImg2017_11_02_15_01_37trained",
        "/Volumes/CPSC587DATA/RecordedImg2017_11_02_15_08_33trained",
    ]
    test_img_folder_list = [
        "/Volumes/CPSC587DATA/RecordedImg20171102152944trained",
        "/Volumes/CPSC587DATA/RecordedImg20171102154113trained",
    ]
    for folder in train_img_folder_list:
        train(m, folder, n_epoch=10)

    for folder in test_img_folder_list:
        test(m, folder)
