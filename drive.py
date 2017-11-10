import argparse
import base64
import os
import random
import shutil
import traceback
from io import BytesIO

import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask

import position
import q_learning_model
from action import get_control_value, ACTION_NUM
from data_utils import process_image

sio = socketio.Server()
app = Flask(__name__)

FRAME_PER_ACTION = 1
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.2  # starting value of epsilon
EXPLORE = 3000000.  # frames over which to anneal epsilon

t = 0
epsilon = INITIAL_EPSILON

replay_memory = None
# cpd_model = None
pm = None
qlm = None


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        try:
            global t, epsilon
            img_orig = Image.open(BytesIO(base64.b64decode(data["image"])))
            img = process_image(img_orig)

            state_img = np.expand_dims(img, 0)
            state_info = np.array([[steering_angle, throttle, speed]])
            state = [state_img, state_info]

            pos_out, act_out = qlm.predict(state)
            pos_idx = np.argmax(pos_out, axis=1)[0]
            reward = position.compute_reward(pos_idx, speed)

            if random.random() <= epsilon:
                action = random.randrange(ACTION_NUM)
                msg = "{:6} EXPLORING {:3} Random Action:    {}->{}"
            else:
                action = np.argmax(act_out, axis=1)[0]
                msg = "{:6} TESTING   {:3} Predict Action:   {}->{}"

            # We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            replay_memory.memorize(reward, img_orig, state, pos_idx, action)

            replay_memory.store_mini_batch()
            # q_learning_model.train_model(qlm, replay_memory)

            # if t % 1000 == 0:
            #     print("Now we save model")
            #     qlm.save_weights(q_learning_model.saved_h5_name, overwrite=True)
            #     with open(q_learning_model.saved_json_name, "w") as outfile:
            #         json.dump(qlm.to_json(), outfile)

            control_value = get_control_value(action)
            print(msg.format(t, position.get_label(pos_idx), action, control_value))

            send_control(*control_value)

            t = t + 1
        except Exception as e:
            print(e)
            traceback.print_exc()
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
        required=True,
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    if args.image_folder != '':
        path = args.image_folder
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        print("RECORDING IMAGES in {} ...".format(path))
    else:
        print("NOT RECORDING IMAGES ...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)

    replay_memory = q_learning_model.ReplayMemory(args.image_folder)

    qlm = q_learning_model.build_model()
    if os.path.exists(q_learning_model.saved_h5_name):
        print('load saved model')
        qlm.load_weights(q_learning_model.saved_h5_name)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
