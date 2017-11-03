import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.utils import to_categorical

import DQN_model
import img_utils
from DQN_model import build_model
from car_position import POS_NUM
from img_utils import process_image


def load_data(img_folder):
    checker = img_utils.ImageChecker(img_folder)
    info = checker.info

    img_list = []
    for _, row in info.iterrows():
        img = Image.open(os.path.join(img_folder, row['filename']))
        img = process_image(img)
        img_list.append(np.expand_dims(img, 0))
    state_img = np.concatenate(tuple(img_list), axis=0)
    state_info = np.asarray(info[['steering_angle', 'throttle', 'speed']])
    print(state_img.shape, state_info.shape)
    return [state_img, state_info], np.asarray(info['car_pos_idx'])


parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    '-i', '--image-folder',
    type=str,
    nargs='+',
    required=True,
    help='Path to image folder.'
)
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

model = build_model()

if os.path.exists(DQN_model.saved_model_name):
    print('load saved model')
    model.load_weights(DQN_model.saved_model_name)

for folder in args.image_folder:
    state, car_pos_idx = load_data(folder)
    print(to_categorical(car_pos_idx, num_classes=POS_NUM))
    for epoch in range(5):
        loss = DQN_model.train_position_part(model, state, car_pos_idx)
        print("Training on {}, data number {}, loss {}".format(folder, state[0].shape[0], loss))
    if os.path.exists(DQN_model.saved_model_name):
        new_model_name = 'DQN_model-{}.h5'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
        shutil.move(DQN_model.saved_model_name, new_model_name)
    print('Save model...')
    model.save_weights(DQN_model.saved_model_name, overwrite=True)
    with open("DQN_model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    new_folder = folder + datetime.now().strftime('%Y%m%d%H%M%S') + 'trained'
    shutil.move(folder, new_folder)
