import json

import tensorflow as tf
from keras import backend as K

from data_utils import data_folders
from q_learning_model import build_model, train_model, saved_h5_name, saved_json_name

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

qlm = build_model()

for epoch in range(1):
    for folder in data_folders:
        print("Train on", folder)
        train_model(qlm, folder)

    print('Save model...')
    qlm.save_weights(saved_h5_name, overwrite=True)
    with open(saved_json_name, "w") as outfile:
        json.dump(qlm.to_json(), outfile)
