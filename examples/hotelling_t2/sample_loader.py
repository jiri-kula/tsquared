#%% /home/jiri/Neural/platform/color/color_a.py

import cv2 as cv
import numpy as np

def load_images():
    base = "/home/jiri/Neural/platform/color/images/"

    red = cv.imread(base + "image_green.png")
    green = cv.imread(base + "image_dark_green.png")

    return red, green

def as_features(image):
    colorspace = cv.COLOR_BGR2LAB
    red_lab = cv.cvtColor(image, colorspace)

    C1 = 1
    C2 = 2

    red_a =   red_lab[:,:,C1]
    red_b =   red_lab[:,:,C2]

    return np.array([red_a.ravel(), red_b.ravel()]).T

def load_dataset():
    red, green = load_images()

    return as_features(red), as_features(green)
# %%
