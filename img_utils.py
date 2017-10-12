import cv2
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3


def process_image(img):
    img_array = np.asarray(img)
    img_array = cv2.resize(img_array, (160, 320), cv2.INTER_AREA)
    img_array = img_array[60:-25, :, :]
    img_array = cv2.resize(img_array, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_array = np.reshape(img_array, (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS))
    return [img_array]
