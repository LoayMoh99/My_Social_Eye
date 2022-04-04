import os
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
import re
import time

def read_gray_from_dir(path: str):
    '''Reads and returns all images in a directory provied as **Path**'''
    imgs = []
    file: str
    pattern = re.compile('.*\.(png|jpeg|jpg)')
    for file in os.listdir(path):
        if(re.search(pattern, file)):
            img_path = os.path.join(path, file)
            imgs.append(read_and_process(img_path))

    return imgs

def read_and_process(img_path):
    ''' Here we apply all kinds of preprocessing for the training images before supplying
        the feature generation models'''
    # Read and convert to Gray
    img = io.imread(img_path, as_gray=True) * 255
    img = img.astype('uint8')
    # Resize ?
    # Crop ?
    
    return img



# t1 = time.time()
# read_gray_from_dir('./Emotions_Detection/test_images')
# print(time.time() - t1)