import os
import json
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K

import DQN_model
from img_utils import process_image

path_list = [
    "/Volumes/CPSC587DATA/TRAINING_IMAGES",
    "/Volumes/CPSC587DATA/RecordedImg2017_11_02_13_43_40trained",
    "/Volumes/CPSC587DATA/RecordedImg2017_11_02_14_03_04trained",
    "/Volumes/CPSC587DATA/RecordedImg2017_11_02_15_01_37trained",
    "/Volumes/CPSC587DATA/RecordedImg2017_11_02_15_08_33trained",
    "/Volumes/CPSC587DATA/RecordedImg20171102152944trained",
    "/Volumes/CPSC587DATA/RecordedImg20171102154113trained",
    "/Volumes/CPSC587DATA/RecordedImg20171102165209trained",
    "/Volumes/CPSC587DATA/RecordedImg20171102172503trained",
    "/Volumes/CPSC587DATA/RecordedImg20171102172849trained",
    "/Volumes/CPSC587DATA/RecordedImg20171102173621trained",
]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

model = DQN_model.build_model(0.0001)

for epoch in range(10):
    for path in path_list:
        csv_path = os.path.join(path, 'info.csv')
        info = pd.read_csv(csv_path)
        info = info.sample(frac=0.1)
        img_list = []
        for _, row in info.iterrows():
            img = Image.open(os.path.join(path, row['filename']))
            img = process_image(img)
            img_list.append(np.expand_dims(img, 0))
        state_img = np.concatenate(tuple(img_list), axis=0)
        state_info = np.asarray(info[['steering_angle', 'throttle', 'speed']])
        state = [state_img, state_info]
        car_pos_idx = np.asarray(info['car_pos_idx'])

        loss = DQN_model.train_position_part(model, state, car_pos_idx)
        print("Training on {}, data number {}, loss {}".format(path, state[0].shape[0], loss))
    print('Save model...')
    model.save_weights(DQN_model.saved_model_name, overwrite=True)
    with open("DQN_model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
