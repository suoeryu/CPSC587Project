import os
import tkinter as tk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

import position

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3


def process_image(img):
    image = np.asarray(img)
    image = cv2.resize(image, (160, 320), cv2.INTER_AREA)
    image = image[60:-25, :, :]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def reduce_dim(img):
    return np.reshape(img, (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS))


class ImageChecker:
    def __init__(self, img_folder) -> None:
        self.folder = img_folder
        self.csv_path = os.path.join(self.folder, 'info.csv')
        self.info = pd.read_csv(self.csv_path)

        self.index = 0
        self.window = tk.Tk()
        self.window.title("{}/{}".format(self.index + 1, self.info.shape[0]))
        self.window.geometry("320x240")
        self.window.configure(background='grey')

        self.img_frame = tk.Frame(self.window, width=320, height=160)
        self.img_frame.pack(side="top")
        img_path = os.path.join(self.folder, self.info['filename'][self.index])
        img = ImageTk.PhotoImage(Image.open(img_path))
        self.img_panel = tk.Label(self.img_frame, image=img)
        self.img_panel.pack(fill="both", expand="yes")

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(side="bottom", fill="both", expand="yes")
        self.prev_button = tk.Button(self.button_frame, text="Prev", command=self.prev_image)
        self.prev_button.pack(side="left")
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.next_image)
        self.next_button.pack(side="right")

        self.pos_var = tk.IntVar()
        self.pos_var.set(self.info['car_pos_idx'][self.index])
        self.radio_frame = tk.Frame(self.window)
        self.radio_frame.pack(side="top")
        for pos_idx, text in enumerate(position.pos_labels):
            radio_button = tk.Radiobutton(self.radio_frame, text=text, variable=self.pos_var,
                                          value=pos_idx, command=self.pos_selection)
            radio_button.pack(side="left")

        self.window.protocol("WM_DELETE_WINDOW", lambda: self.quit())
        self.window.bind('<Escape>', lambda e: self.quit())
        self.window.bind('n', lambda e: self.next_image())
        self.window.bind('p', lambda e: self.prev_image())
        self.window.bind('1', lambda e: self.select_pos(0))
        self.window.bind('2', lambda e: self.select_pos(1))
        self.window.bind('3', lambda e: self.select_pos(2))
        self.window.bind('4', lambda e: self.select_pos(3))
        self.window.bind('s', lambda e: self.save_csv())
        self.window.mainloop()

    def change_image(self, step):
        self.index = self.index + step
        img_path = os.path.join(self.folder, self.info['filename'][self.index])
        img = ImageTk.PhotoImage(Image.open(img_path))
        self.img_panel.configure(image=img)
        self.img_panel.image = img
        self.window.title("{}/{}".format(self.index + 1, self.info.shape[0]))
        self.pos_var.set(self.info['car_pos_idx'][self.index])

    def next_image(self):
        if self.index < self.info.shape[0] - 1:
            self.change_image(1)

    def prev_image(self):
        if self.index > 0:
            self.change_image(-1)

    def pos_selection(self):
        self.info.set_value(self.index, 'car_pos_idx', self.pos_var.get())

    def select_pos(self, idx):
        self.pos_var.set(idx)
        self.pos_selection()

    def save_csv(self):
        print("save info to csv")
        self.info.to_csv(self.csv_path)

    def quit(self):
        self.save_csv()
        self.window.destroy()


if __name__ == '__main__':
    folder_list = [
        '/Volumes/CPSC587DATA/RecordedImg',
    ]
    for folder in folder_list:
        ImageChecker(folder)
