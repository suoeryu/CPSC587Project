import random

import numpy as np

STEERING_ANGLE_NUM = 9
THROTTLE_NUM = 3
ACTION_NUM = STEERING_ANGLE_NUM * THROTTLE_NUM

steering_angle_values = np.linspace(-1, 1, STEERING_ANGLE_NUM)
throttle_values = np.linspace(0, 1, THROTTLE_NUM)


def create_action(steering_angle, throttle):
    def f(x, n): return int(round((x + 1) / 2 * (n - 1)))

    steering_angle_index = f(steering_angle, STEERING_ANGLE_NUM)
    throttle_index = f(throttle, THROTTLE_NUM)
    return steering_angle_index * THROTTLE_NUM + throttle_index


def get_control_value(action):
    return steering_angle_values[int(action / THROTTLE_NUM)], throttle_values[action % THROTTLE_NUM]


def get_random_action():
    return random.randint(0, ACTION_NUM - 1)


if __name__ == '__main__':
    for a in range(ACTION_NUM):
        print(a, get_control_value(a))
