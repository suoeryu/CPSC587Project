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
model.load_weights(position_model.saved_h5_name)

for folder in position_model.test_list:
    print("Evaluate model on ", folder)
    X, y = load_data(folder)
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(X), axis=1)
    print(confusion_matrix(y_true, y_pred))
    print('Precision:\t', precision_score(y_true, y_pred, average=None))
    print('Recall:\t', recall_score(y_true, y_pred, average=None))
    print('F1 score:\t', f1_score(y_true, y_pred, average=None))
print('Save model...')
model.save_weights(position_model.saved_h5_name, overwrite=True)
with open(position_model.saved_json_name, "w") as outfile:
    json.dump(model.to_json(), outfile)
