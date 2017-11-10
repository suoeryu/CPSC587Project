import os
import numpy as np
import tensorflow as tf
from keras import backend as K

from data_utils import load_data
from q_learning_model import build_model, train_model, saved_h5_name, saved_json_name

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

qlm = build_model()

# if os.path.exists(saved_h5_name):
#     print('load q learning model')
#     qlm.load_weights(saved_h5_name)

data_folder = '/Volumes/CPSC587DATA/RecordedImg'
info, img_map = load_data(data_folder)

info = info[['cur_filename', 'cur_steering_angle', 'cur_throttle', 'cur_speed']].drop_duplicates()
# print(info)
input_img = np.concatenate([i for i in map(lambda x: img_map[x], info['cur_filename'])], axis=0)
input_info = np.asarray(info[['cur_steering_angle', 'cur_throttle', 'cur_speed']])

pos, q_sa = qlm.predict([input_img, input_info])
print(pos)
for i in range(q_sa.shape[0]):
    print(q_sa[i])
