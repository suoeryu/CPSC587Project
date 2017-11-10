import csv
import random
import os
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, concatenate, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical

import position
from action import ACTION_NUM
from position import POS_NUM
from data_utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, load_data

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99

reward_dict = {'ON': 0.5, 'LEFT': -0.5, 'RIGHT': -0.5, 'OFF': -1}

saved_h5_name = 'q_learning_model.h5'
saved_json_name = 'q_learning_model.json'


class ReplayMemory:
    def __init__(self, store_folder, max_num=50000) -> None:
        self.csv_fieldnames = ['cur_filename', 'cur_pos_idx',
                               'cur_steering_angle', 'cur_throttle', 'cur_speed',
                               'action', 'reward', 'next_filename', 'next_pos_idx',
                               'next_steering_angle', 'next_throttle', 'next_speed']
        self.memory = deque()
        self.max_num = max_num
        self.store_folder = store_folder
        if store_folder:
            self.csv_path = os.path.join(store_folder, 'info.csv')
            with open(self.csv_path, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
                writer.writeheader()
        self.thread_pool = ThreadPoolExecutor(4)

        self.state = None
        self.filename = None
        self.pos_idx = None
        self.action = None
        self.reward = None
        self.next_state = None

    def __len__(self) -> int:
        return len(self.memory)

    def memorize(self, reward, img_orig, state, pos_idx, action) -> None:
        # print(state)
        img_filename = self.save_img(img_orig)
        self.reward = reward
        self.next_state = state
        if self.state is not None and self.action is not None:
            cur_state_info = np.squeeze(self.state[1])
            next_state_info = np.squeeze(self.next_state[1])
            self.memory.append(
                {"cur_filename": self.filename,
                 # "cur_state": self.state,
                 "cur_pos_idx": self.pos_idx,
                 'cur_steering_angle': cur_state_info[0],
                 'cur_throttle': cur_state_info[1],
                 'cur_speed': cur_state_info[2],
                 "action": self.action,
                 "reward": self.reward,
                 "next_filename": img_filename,
                 "next_pos_idx": pos_idx,
                 "next_steering_angle": next_state_info[0],
                 'next_throttle': next_state_info[1],
                 'next_speed': next_state_info[2],
                 # "next_state": self.next_state,
                 })
            if len(self.memory) > self.max_num:
                self.memory.popleft()
        self.filename = img_filename
        self.state = state
        self.pos_idx = pos_idx
        self.action = action

    def get_mini_batch(self, k=BATCH_SIZE):
        return random.sample(self.memory, k) if len(self.memory) > k * 4 else None

    def store_mini_batch(self, k=BATCH_SIZE):
        samples = random.sample(self.memory, k) if len(self.memory) > k * 4 else None
        if samples is not None:
            with open(self.csv_path, 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
                csv_writer.writerows(samples)

    def save_img(self, img_orig):
        if self.store_folder:
            filename = '{}.jpg'.format(datetime.now().strftime('%Y%m%d%H%M%S.%f')[:-3])
            full_path = os.path.join(self.store_folder, filename)
            self.thread_pool.submit(lambda: img_orig.save(full_path))
            return filename
        else:
            return None


def build_model(learn_rate=LEARNING_RATE):
    print("Now we build the Q learning model")
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    img_values = input_img

    conv_layer_1 = Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu')
    img_values = conv_layer_1(img_values)
    img_values = BatchNormalization()(img_values)

    conv_layer_2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')
    img_values = conv_layer_2(img_values)
    img_values = BatchNormalization()(img_values)

    conv_layer_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    img_values = conv_layer_3(img_values)
    img_values = BatchNormalization()(img_values)

    max_poll_layer = MaxPooling2D(pool_size=(2, 2))
    img_values = max_poll_layer(img_values)

    img_values = Flatten()(img_values)

    pos_values = BatchNormalization()(img_values)
    pos_hidden_layer = Dense(64, activation='relu')
    pos_values = pos_hidden_layer(pos_values)

    pos_output_layer = Dense(POS_NUM, activation='relu')
    output_pos = pos_output_layer(pos_values)

    input_info = Input(shape=(3,))
    info_values = BatchNormalization()(input_info)

    act_values = concatenate([img_values, info_values], axis=-1)
    act_values = BatchNormalization()(act_values)

    act_hidden_layer = Dense(512, activation='relu')
    act_values = act_hidden_layer(act_values)

    act_output_layer = Dense(ACTION_NUM, activation='relu')
    output_act = act_output_layer(act_values)

    model = Model(inputs=[input_img, input_info], outputs=[output_pos, output_act])

    adam = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=adam)

    print("We finish building the Q learning model")
    # model.summary()
    return model


def train_model(model, data_folder, batch_size=BATCH_SIZE, check=False):
    info, img_map = load_data(data_folder, check)
    n_rows = info.shape[0]
    for start in range(0, n_rows, batch_size):
        end = start + batch_size
        if end > n_rows:
            end = n_rows
        sl = slice(start, end)
        file_names = info['cur_filename'][sl]
        input_img = np.concatenate([i for i in map(lambda x: img_map[x], file_names)], axis=0)
        input_info = np.asarray(info[['cur_steering_angle', 'cur_throttle', 'cur_speed']][sl])
        state = [input_img, input_info]
        output_pos = to_categorical(info['cur_pos_idx'][sl], num_classes=POS_NUM)
        _, output_act = model.predict([input_img, input_info])

        next_imgs = [i for i in map(lambda x: img_map[x], info['next_filename'][sl])]
        next_infos = np.asarray(info[['next_steering_angle', 'next_throttle', 'next_speed']][sl])
        for index, row in info[['cur_pos_idx', 'action', 'reward']][sl].iterrows():
            index = index - start
            action = int(row['action'])
            pos = position.get_label(int(row['cur_pos_idx']))
            reward = row['reward']
            if pos == 'OFF':
                output_act[index, action] = reward
            else:
                next_state = [next_imgs[index], next_infos[index:index + 1]]
                _, q_sa = model.predict(next_state)
                output_act[index, action] = reward + GAMMA * np.max(q_sa)
        output = [output_pos, output_act]
        loss = model.train_on_batch(state, output)
        print("Iter:{:4} Loss: {:.6f}, pos loss {:.6f} act loss {:.6f}".format(
            int(start / batch_size), *loss))
