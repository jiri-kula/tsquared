#%% /home/jiri/Neural/platform/color/color_a.py

import cv2 as cv
import numpy as np
import glob

def load_images():
    file_extension = "*.png"
    base = "/home/jiri/Neural/platform/color/images/"

    files = glob.glob(base + file_extension)

    images = []
    for file in files:
        images.append(cv.imread(file))

    return images

def as_features(image):
    colorspace = cv.COLOR_BGR2LAB
    red_lab = cv.cvtColor(image, colorspace)

    C1 = 1
    C2 = 2

    red_a =   red_lab[:,:,C1]
    red_b =   red_lab[:,:,C2]

    return np.array([red_a.ravel(), red_b.ravel()]).T

def load_dataset():
    images = load_images()

    features = []

    for image in images:
        features.append(as_features(image))

    return features
# %%
