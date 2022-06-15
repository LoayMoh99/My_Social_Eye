import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np
import cv2


class LocalBinaryPatterns:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.num_points, self.radius)
        hist = plt.hist(lbp.ravel())
        return hist


# start our describer LBP
desc = LocalBinaryPatterns(8, 4)


def extractFaceTrackFeature(face_img):
    # preprocessing by change it to gray scale
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # extract LBP feature
    lbp_hist = desc.describe(face)
    return np.array(lbp_hist[0])
