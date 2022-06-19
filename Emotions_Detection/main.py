

# from Assets.CommonFuntions import *
# from Models.features import lpq, lpq_plus, LPQ, LPQPlus, PHOG_Algorithm
from cv2 import imread
import sklearn
import cv2
import skimage.io as io
from skimage.color import rgb2gray
from sklearn.svm import SVC
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from Emotions_Detection.features import LPQ, PHOG_Algorithm

current = os.path.dirname(os.path.realpath(os.getcwd()))
parent = os.path.dirname(current)
sys.path.append(parent)


# 1- Read the Image(s)


class EmotionDetectionModel:
    def __init__(self, model_file='lpq_phog_model.sav', feats='lpq_phog', is_prob=True) -> None:
        self.lpq = LPQ()
        file_path = os.path.join(os.getcwd(), 'Emotions_detection')
        file_path = os.path.join(file_path, model_file)
        self.model = pickle.load(open(file_path, 'rb'))
        self.is_prob = is_prob
        self.feats = feats

    def get_labels(self, img, frame=None):

        # Get and process Face
        x, y, w, h = frame
        face = img[y:y+h, x:x+w]
        face_gray = rgb2gray(face)

        # img = self.process_image(face_gray)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        face_gray = self.process_image(face_gray)
        features = []
        phog_desc = None

        if 'lpq' in self.feats:
            # get LPQ Feature
            features = self.lpq.compute(face_gray)

        if 'phog' in self.feats:
            # get PHOG Feature
            phog_desc = PHOG_Algorithm(face_gray)
            features = np.concatenate((features, phog_desc))

        # if self.feats == 'lpq_phog':
        #     # concatenate the two features
        #     features = np.concatenate((features, phog_desc))
        if not self.is_prob:
            # Predict Emotions String
            pred = self.model.predict([features])

            return pred[0]
        else:
            # Predict Emotions Propabilities
            pred = self.model.predict_proba([features])

            return pred[0]

    def process_image(self, img):
        img = ((img)*255).astype('uint8')
        if len(img.shape) > 2:
            img = rgb2gray(img)
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_NEAREST)
        return img


# # 2- Extract Faces from image
# faces = get_faces_from_image(img, is_dir=False)
# x, y, w, h = faces[0]
# roi = img[y:y+h, x:x+w]


# # 3- Get LPQ feature from all faces extracted
# LPQ_desc = lpq()
