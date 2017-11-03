import argparse
import base64
import csv
import json
import random
from datetime import datetime
import os
import shutil

import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

import DQN_model
import car_position
from action import generate_hardcoded_action, get_control_value, ACTION_NUM
from img_utils import process_image

sio = socketio.Server()
app = Flask(__name__)

FRAME_PER_ACTION = 1
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
EXPLORE = 3000000.  # frames over which to anneal epsilon
OBSERVE = DQN_model.OBSERVATION

t = 0
epsilon = INITIAL_EPSILON

replay_memory = DQN_model.ReplayMemory()
dqn_model = None
cpd_model = None

csv_fieldnames = ['filename', 'car_pos_idx', 'steering_angle', 'throttle', 'speed']


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # print(data)
        try:
            global t, epsilon
            img_orig = Image.open(BytesIO(base64.b64decode(data["image"])))
            img = process_image(img_orig)
            # car_pos = cpd_model.predict([reduce_dim(img)])[0]
            # car_pos_idx = car_position.get_index(car_pos)

            state_img = np.expand_dims(img, 0)
            state_info = np.array([[steering_angle, throttle, speed]])
            state = [state_img, state_info]

            car_pos_one_hot, q = dqn_model.predict(state)
            dqn_pos_idx = np.argmax(car_pos_one_hot, axis=1)[0]
            # print(car_pos_one_hot, dqn_pos_idx)
            car_pos = car_position.get_label(dqn_pos_idx)

            reward = DQN_model.compute_reward(car_pos, speed)

            if t < OBSERVE:  # observing, using hardcoded action
                action = generate_hardcoded_action(car_pos, speed)
                msg = "{:6} OBSERVING {:5} Hardcoded Action: {}->{}"
            else:
                if random.random() <= epsilon:
                    action = random.randrange(ACTION_NUM)
                    msg = "{:6} EXPLORING {:5} Random Action:    {}->{}"
                else:
                    action = np.argmax(q)
                    msg = "{:6} TRAINING  {:5} Predict Action:   {}->{}"

                # We reduced the epsilon gradually
                if epsilon > FINAL_EPSILON and t > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                    # if replay_memory.has_enough_sample() and t % 10 == 0:
                    # mini_batch = replay_memory.get_mini_batch()
                    # batch_train_model(qlm, mini_batch)

                    # if t % 1000 == 0:
                    #     print("Now we save model")
                    #     qlm.save_weights("q_learning_model.h5", overwrite=True)
                    #     with open("q_learning_model.json", "w") as outfile:
                    #         json.dump(qlm.to_json(), outfile)

            replay_memory.memorize(reward, state, car_position.get_index(car_pos), action)

            control_value = get_control_value(action)
            print(msg.format(t, car_pos, action, control_value))

            send_control(*control_value)

            # save images
            if args.image_folder != '' and t % 5 == 0:
                folder_path = args.image_folder
                filename = '{}.jpg'.format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3])
                img_orig.save(os.path.join(folder_path, filename))
                with open(os.path.join(path, 'info.csv'), 'a') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
                    csv_writer.writerow({'filename': filename, 'car_pos_idx': dqn_pos_idx,
                                         'steering_angle': steering_angle, 'throttle': throttle,
                                         'speed': speed})

            t = t + 1
        except Exception as e:
            print(e)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '-m', '--mode',
        type=str,
        nargs='?',
        default='train',
        help='Run model, can be train, run, and continue'
    )
    parser.add_argument(
        '-s', '--image-folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    if args.image_folder != '':
        path = args.image_folder
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        with open(os.path.join(path, 'info.csv'), 'w') as file:
            writer = csv.DictWriter(file, fieldnames=csv_fieldnames)
            writer.writeheader()
        print("RECORDING IMAGES in {} ...".format(path))
    else:
        print("NOT RECORDING IMAGES ...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)
    cpd_model = joblib.load('car_pos_detection.pkl')
    dqn_model = DQN_model.build_model()
    if os.path.exists(DQN_model.saved_model_name):
        print('load saved model')
        dqn_model.load_weights(DQN_model.saved_model_name)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
