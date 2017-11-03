import random
import os
from collections import deque

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical

import car_position
from action import ACTION_NUM
from car_position import POS_NUM
from img_utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99

OBSERVATION = 3000

reward_dict = {'ON': 0.5, 'LEFT': -0.5, 'RIGHT': -0.5, 'OFF': -1}

saved_model_name = 'DQN_model.h5'


class ReplayMemory:
    def __init__(self, max_num=50000) -> None:
        self.memory = deque()
        self.max_num = max_num

        self.state = None
        self.state_filename = None
        self.car_pos_index = None
        self.action = None
        self.reward = None
        self.next_state = None

    def __len__(self) -> int:
        return len(self.memory)

    def memorize(self, reward, state, car_pos_index, action) -> None:
        self.reward = reward
        self.next_state = state
        if self.state is not None and self.action is not None:
            self.memory.append(
                (self.state, self.car_pos_index, self.action, self.reward, self.next_state))
            if len(self.memory) > self.max_num:
                self.memory.popleft()
        self.state = state
        self.car_pos_index = car_pos_index
        self.action = action

    def get_mini_batch(self, k=BATCH_SIZE):
        return random.sample(self.memory, k) if len(self.memory) > k else None


def compute_reward(car_pos, speed):
    reward = reward_dict[car_pos]
    reward = reward + speed / 100
    return reward


def build_model(learn_rate=LEARNING_RATE):
    print("Now we build the model")
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    img_values = input_img

    conv_layer_1 = Conv2D(32, (10, 10), strides=(3, 10), padding='same', activation='relu')
    img_values = conv_layer_1(img_values)

    conv_layer_2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')
    img_values = conv_layer_2(img_values)

    conv_layer_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    img_values = conv_layer_3(img_values)

    img_values = Flatten()(img_values)
    img_values = BatchNormalization()(img_values)

    pos_hidden_layer = Dense(64, activation='relu')
    pos_values = pos_hidden_layer(img_values)

    pos_output_layer = Dense(POS_NUM, activation='relu')
    output_pos = pos_output_layer(pos_values)

    input_info = Input(shape=(3,))
    info_norm = BatchNormalization()(input_info)

    comb = concatenate([img_values, info_norm], axis=-1)

    action_hidden_layer = Dense(512, activation='relu')
    action_values = action_hidden_layer(comb)

    action_output_layer = Dense(ACTION_NUM, activation='softmax')
    output_act = action_output_layer(action_values)

    model = Model(inputs=[input_img, input_info], outputs=[output_pos, output_act])

    adam = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, loss_weights=[1., .9])

    print("We finish building the model")
    # model.summary()
    return model


def train_action_part(model, replay_memory, batch_size=BATCH_SIZE):
    mini_batch = replay_memory.get_mini_batch()
    if mini_batch is not None:
        inputs_img = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        inputs_info = np.zeros((batch_size, 3))
        outputs_pos = np.zeros(batch_size, POS_NUM)
        outputs_act = np.zeros((batch_size, ACTION_NUM))
        for i in range(batch_size):
            state = mini_batch[i][0]
            car_pos_index = mini_batch[i][2]
            action = mini_batch[i][2]
            reward = mini_batch[i][3]
            next_state = mini_batch[i][4]
            inputs_img[i:i + 1] = state[0]
            inputs_info[i:i + 1] = state[1]
            outputs_pos[i][car_pos_index] = 1
            outputs_act[i] = model.predict(state)
            car_pos = car_position.get_label(car_pos_index)
            if car_pos == 'OFF':
                outputs_act[i, action] = reward
            else:
                _, q_sa = model.predict(next_state)
                outputs_act[i, action] = reward + GAMMA * np.max(q_sa)
        inputs = [inputs_img, inputs_info]
        outputs = [outputs_pos, outputs_act]
        return model.train_on_batch(inputs, outputs)
    else:
        return None


def train_position_part(model, states, pos_index):
    _, outputs_act = model.predict(states)
    outputs_pos = to_categorical(pos_index, num_classes=POS_NUM)
    return model.train_on_batch(states, [outputs_pos, outputs_act])


