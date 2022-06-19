from cv2 import imread
import sklearn
from Models.features import lpq, lpq_plus, LPQ, LPQPlus, PHOG_Algorithm
from Assets.CommonFuntions import *

import cv2
import skimage.io as io
from skimage.color import rgb2gray
from sklearn.svm import SVC

import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1- Read the Image(s)
class EmotionDetectionModel:
    def __init__(self, model_file='lpq_phog_model.sav', feats='lpq_phog', is_prob = True) -> None:
        self.lpq = LPQ()
        file_path = os.path.join(os.getcwd(), 'Emotions_detection')
        file_path = os.path.join(file_path, model_file)
        self.model:SVC  = pickle.load(open(file_path, 'rb'))
        self.is_prob = is_prob
        self.feats = feats

    def get_labels_prob(self, img, frame=None):
        if self.is_prob:
            # Get and process Face
            x, y, w, h = frame
            face = img[y:y+h, x:x+w]
            face_gray = rgb2gray(img)
            lpq_desc = None
            phog_desc = None

            if 'lpq' in self.feats:
                # get LPQ Feature
                lpq_desc = self.lpq.compute(face_gray)

            if 'phog' in self.feats:
                # get PHOG Feature
                phog_desc = PHOG_Algorithm(face_gray)

            if self.feats == 'lpq_phog':
                # concatenate the two features
                features = np.concatenate((lpq_desc, phog_desc), axis=1)
                    
            # Predict Emotions Propabilities
            pred = self.model.predict([features])

            return pred
        return None 

    def get_labels(self, img, frame=None):
        if not self.is_prob:
            # Get and process Face
            x, y, w, h = frame
            face = img[y:y+h, x:x+w]            
            face = self.process_image(face)
            face_gray = img
            lpq_desc = None
            phog_desc = None

            if 'lpq' in self.feats:
                # get LPQ Feature
                lpq_desc = self.lpq.compute(face_gray)

            if 'phog' in self.feats:
                # get PHOG Feature
                phog_desc = PHOG_Algorithm(face_gray)

            if self.feats == 'lpq_phog':
                # concatenate the two features
                features = np.concatenate((lpq_desc, phog_desc))
                    
            # Predict Emotions Propabilities
            pred = self.model.predict([features])

            return pred
        return None
    
    def process_image(self, img):
        img = img.astype('uint8')
        if len(img.shape) >2:
            img = rgb2gray(img)
        img = cv2.resize(img, (48,48), interpolation = cv2.INTER_NEAREST)
        return img


        

# # 2- Extract Faces from image
# faces = get_faces_from_image(img, is_dir=False)
# x, y, w, h = faces[0]
# roi = img[y:y+h, x:x+w]


# # 3- Get LPQ feature from all faces extracted
# LPQ_desc = lpq()