import os
import json
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

import position_model
from data_utils import load_data
from img_utils import process_image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

model = position_model.build_model()

for epoch in range(100):
    print("Epoch: ", epoch)
    for folder in position_model.train_list:
        X, y = load_data(folder)
        loss = model.train_on_batch(X, y)
        print("Training on {}, data number {}, loss {}".format(folder, X.shape[0], loss))

print('Save model...')
model.save_weights(position_model.saved_h5_name, overwrite=True)
with open(position_model.saved_json_name, "w") as outfile:
    json.dump(model.to_json(), outfile)
