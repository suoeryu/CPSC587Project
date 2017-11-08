import json
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.utils import to_categorical

import img_utils
import position_model
from data_utils import load_data
from position import POS_NUM
from img_utils import process_image
from position_model import build_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

model = build_model()
saved_file_name = position_model.saved_h5_name

if os.path.exists(saved_file_name):
    print('load saved model')
    model.load_weights(saved_file_name)

record_folder = '/Volumes/CPSC587DATA/RecordedImg'
X, y = load_data(record_folder, check=True)

for epoch in range(10):
    loss = model.train_on_batch(X, y)
    print("Training on {}, data number {}, loss {}".format(record_folder, X.shape[0], loss))

if os.path.exists(saved_file_name):
    new_model_name = '{}-{}'.format(position_model.saved_h5_name,
                                    datetime.now().strftime('%Y%m%d%H%M%S'))
    shutil.move(saved_file_name, new_model_name)
model.save_weights(saved_file_name, overwrite=True)
print('Save model...')
with open(position_model.saved_json_name, "w") as outfile:
    json.dump(model.to_json(), outfile)

new_folder = record_folder + datetime.now().strftime('%Y%m%d%H%M%S')
print('mover {} to {}'.format(record_folder, new_folder))
shutil.move(record_folder, new_folder)
