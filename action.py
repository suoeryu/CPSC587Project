import random

import numpy as np

STEERING_ANGLE_NUM = 9
THROTTLE_NUM = 5
ACTION_NUM = STEERING_ANGLE_NUM * THROTTLE_NUM

steering_angle_values = np.linspace(-1, 1, STEERING_ANGLE_NUM)
throttle_values = np.linspace(-1, 1, THROTTLE_NUM)


def create_action(steering_angle, throttle):
    def f(x, n): return int(round((x + 1) / 2 * (n - 1)))

    steering_angle_index = f(steering_angle, STEERING_ANGLE_NUM)
    throttle_index = f(throttle, THROTTLE_NUM)
    return steering_angle_index * THROTTLE_NUM + throttle_index


def get_control_value(action):
    return steering_angle_values[int(action / THROTTLE_NUM)], \
           throttle_values[action % THROTTLE_NUM]


def get_random_action():
    return random.randint(0, ACTION_NUM - 1)


steer_option = {
    'LEFT': lambda: random.uniform(0.7, 1),
    'RIGHT': lambda: random.uniform(-1, -0.7),
    'ON': lambda: random.uniform(-0.1, 0.1),
    'OFF': lambda: random.uniform(-1., 1.)
}

throttle_option = {
    'LEFT': lambda speed: (15 - speed) / 30,
    'RIGHT': lambda speed: (15 - speed) / 30,
    'ON': lambda speed: (30 - speed) / 30,
    'OFF': lambda speed: (10 - speed) / 30
}


def generate_hardcoded_action(car_pos, speed):
    steering_angle = steer_option[car_pos]()
    throttle = throttle_option[car_pos](speed)
    return create_action(steering_angle, throttle)


if __name__ == '__main__':
    for a in range(10):
        print(a, get_control_value(a))
