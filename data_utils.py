import csv
import os
from os.path import isfile, join

import numpy as np
import pandas as pd
from PIL import Image

from DQN_model import pos_labels
from img_utils import process_image, reduce_dim

root_path = "/Volumes/CPSC587DATA/TRAINING_IMAGES"


def create_index_csv():
    with open(join(root_path, 'index.csv'), 'w') as csv_file:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for label in pos_labels:
            path = join(root_path, label)
            jpg_files = [join(path, f) for f in os.listdir(path) if
                         isfile(join(path, f)) and f.endswith('.jpg')]
            for jpg in jpg_files:
                writer.writerow({'filename': jpg, 'label': label})


def load_image_data():
    csv_path = join(root_path, "index.csv")
    info = pd.read_csv(csv_path)
    img_array_list = []
    for file in info['filename']:
        img = Image.open(file)
        img = process_image(img)
        img_array_list.append([reduce_dim(img)])
    img_data = np.concatenate(tuple(img_array_list), axis=0)
    labels = np.array(info['label'])
    return img_data, labels


def split_train_test(data, labels, test_ratio=0.3):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    X_train = data[train_indices]
    y_train = labels[train_indices]

    X_test = data[test_indices]
    y_test = labels[test_indices]

    return X_train, y_train, X_test, y_test

