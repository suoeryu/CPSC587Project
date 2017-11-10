import json
import os
import shutil
from datetime import datetime

import tensorflow as tf
from keras import backend as K

from q_learning_model import build_model, saved_h5_name, train_model, saved_json_name

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

qlm = build_model()
if os.path.exists(saved_h5_name):
    print('load q learning model')
    qlm.load_weights(saved_h5_name)

record_folder = '/Volumes/CPSC587DATA/RecordedImg'
train_model(qlm, record_folder, check=True)

if os.path.exists(saved_h5_name):
    new_model_name = '{}-{}'.format(saved_h5_name,
                                    datetime.now().strftime('%Y%m%d%H%M%S'))
    shutil.move(saved_h5_name, new_model_name)
qlm.save_weights(saved_h5_name, overwrite=True)
print('Save model...')
with open(saved_json_name, "w") as outfile:
    json.dump(qlm.to_json(), outfile)

new_folder = record_folder + datetime.now().strftime('%Y%m%d%H%M%S')
print('mover {} to {}'.format(record_folder, new_folder))
shutil.move(record_folder, new_folder)
