#%% /home/jiri/Neural/platform/color/color_a.py

import cv2 as cv
import numpy as np
import glob

def list_directories(base):
    # Append a slash if not present at the end of the base path
    if base[-1] != "/":
        base += "/"

    # Use glob to list all directories in the base directory
    directories = glob.glob(base + "*/")

    # Remove the base path from the directories
    directories = [dir.replace(base, "") for dir in directories]

    return directories

def load_images(base):
    file_extension = "*.png"
    # base = "/mnt/c/tmp/output"

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

def load_dataset(base):
    dirs = list_directories(base)
    
    features = []

    for dir in dirs:
        print("Loading images from " + dir)
        images = load_images(base + "/" + dir)

        image_set_feats = []
        for image in images:
            image_set_feats.append(as_features(image))


        features.append(np.concatenate(image_set_feats))

    # remove trailing slash
    dirs = [dir[:-1] for dir in dirs]

    return features, dirs
# %%
