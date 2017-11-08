import csv
import random
import os
import threading
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
from img_utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99

OBSERVATION = 10000

reward_dict = {'ON': 0.5, 'LEFT': -0.5, 'RIGHT': -0.5, 'OFF': -1}

saved_h5_name = 'q_learning_model.h5'
saved_json_name = 'q_learning_model.json'

csv_fieldnames = ['filename', 'car_pos_idx', 'steering_angle', 'throttle', 'speed', 'action',
                  'reward', 'next_filename']


class ReplayMemory:
    def __init__(self, store_folder, max_num=50000) -> None:
        self.memory = deque()
        self.max_num = max_num
        self.store_folder = store_folder
        if store_folder:
            self.csv_path = os.path.join(store_folder, 'info.csv')
            with open(self.csv_path, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
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
            self.memory.append(
                (self.state, self.pos_idx, self.action, self.reward, self.next_state))
            if self.store_folder:
                with open(self.csv_path, 'a') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
                    csv_writer.writerow({'filename': self.filename,
                                         'car_pos_idx': self.pos_idx,
                                         'steering_angle': state[1][0][0],
                                         'throttle': state[1][0][1],
                                         'speed': state[1][0][2],
                                         'action': action,
                                         'next_filename': img_filename,
                                         'reward': reward,
                                         })
            if len(self.memory) > self.max_num:
                self.memory.popleft()
        self.state = state
        self.filename = img_filename
        self.pos_idx = pos_idx
        self.action = action

    def get_mini_batch(self, k=BATCH_SIZE):
        return random.sample(self.memory, k) if len(self.memory) > k else None

    def save_img(self, img_orig):
        if self.store_folder:
            filename = '{}.jpg'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3])
            full_path = os.path.join(self.store_folder, filename)
            self.thread_pool.submit(lambda: img_orig.save(full_path))
            # img_orig.save(os.path.join(self.store_folder, filename))
            return filename
        else:
            return None


def compute_reward(car_pos, speed):
    reward = reward_dict[car_pos]
    reward = reward + speed / 100
    return reward


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

    input_info = Input(shape=(3,))
    info_values = input_info

    values_comb = concatenate([img_values, info_values], axis=-1)
    values = BatchNormalization()(values_comb)

    hidden_layer = Dense(512, activation='relu')
    values = hidden_layer(values)

    output_layer = Dense(ACTION_NUM, activation='relu')
    output = output_layer(values)

    model = Model(inputs=[input_img, input_info], outputs=output)

    adam = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=adam)

    print("We finish building the Q learning model")
    # model.summary()
    return model


lock = threading.Lock()


def train_model(model, replay_memory, batch_size=BATCH_SIZE):
    mini_batch = replay_memory.get_mini_batch()
    if mini_batch is not None:
        t1 = datetime.now()
        inputs_img = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        inputs_info = np.zeros((batch_size, 3))
        outputs = np.zeros((batch_size, ACTION_NUM))
        for i in range(batch_size):
            state = mini_batch[i][0]
            pos_idx = mini_batch[i][1]
            action = mini_batch[i][2]
            reward = mini_batch[i][3]
            next_state = mini_batch[i][4]
            inputs_img[i:i + 1] = state[0]
            inputs_info[i:i + 1] = state[1]
            car_pos = position.get_label(pos_idx)
            if car_pos == 'OFF':
                outputs[i, action] = reward
            else:
                q_sa = model.predict(next_state)
                outputs[i, action] = reward + GAMMA * np.max(q_sa)
        inputs = [inputs_img, inputs_info]
        t2 = datetime.now()
        with lock:
            loss = model.train_on_batch(inputs, outputs)
        t3 = datetime.now()
        print("Prepare time: {}".format(t2 - t1))
        print("Train time: {}".format(t3 - t2))
