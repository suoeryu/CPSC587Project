import csv
import os
from os.path import isfile, join

import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import to_categorical

import img_utils
from img_utils import process_image, reduce_dim
from position import POS_NUM


def load_data(img_folder, check=False):
    if check:
        checker = img_utils.ImageChecker(img_folder)
        info = checker.info
    else:
        csv_path = os.path.join(img_folder, 'info.csv')
        info = pd.read_csv(csv_path)
    img_list = []
    for _, row in info.iterrows():
        img = Image.open(os.path.join(img_folder, row['filename']))
        img = process_image(img)
        img_list.append(np.expand_dims(img, 0))
    # state_info = np.asarray(info[['steering_angle', 'throttle', 'speed']])
    return (np.concatenate(tuple(img_list), axis=0),
            to_categorical(np.asarray(info['car_pos_idx']), num_classes=POS_NUM))
