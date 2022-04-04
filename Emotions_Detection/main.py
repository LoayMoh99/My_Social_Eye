from Models.face_detection import get_faces_from_image
from Models.LPQ import lpq
from Models.LPQExtended import lpq_plus
from Assets.CommonFuntions import *
import cv2
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

# 1- Read the Image(s)
img = read_and_process('./Emotions_Detection/test_images/img6.jpg') 

# 2- Extract Faces
faces = get_faces_from_image(img, is_dir=False)
x, y, w, h = faces[0]
roi = img[y:y+h, x:x+w]

# 3- Get LPQ feature from all faces extracted
# LPQ_desq = lpq(roi,winSize=11, mode='h')
LPQ_plus_desq = lpq_plus(roi,winSize=11, mode='h')
