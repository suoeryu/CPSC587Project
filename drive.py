import argparse
import base64
from datetime import datetime
import os
import shutil
import random

import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from sklearn.externals import joblib

from img_utils import process_image

sio = socketio.Server()
app = Flask(__name__)
pipe_line = joblib.load("car_detection.pkl")

steer_option = {'OFF_LEFT': lambda: random.uniform(0.5, 1),
                'OFF_RIGHT': lambda: random.uniform(-1, -0.5),
                'ON_ROAD': lambda: random.uniform(-0.1, 0.1)}

throttle_option = {
    'OFF_LEFT': lambda speed: -(speed - 10) / 30 if speed > 10 else (10 - speed) / 10,
    'OFF_RIGHT': lambda speed: -(speed - 10) / 30 if speed > 10 else (10 - speed) / 10,
    'ON_ROAD': lambda speed: (30-speed) / 30
}


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


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
        # print(steering_angle, throttle, speed)
        img = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            # steering_angle = steering_angle + random.uniform(-.1, .1)
            img_array = process_image(img)
            car_pos = pipe_line.predict(img_array)[0]
            steering_angle = steer_option[car_pos]()
            throttle = throttle_option[car_pos](speed)
            print(car_pos, steering_angle, throttle)
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            img.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='',
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
