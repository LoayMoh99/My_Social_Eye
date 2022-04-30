import cv2
import numpy as np
import skimage.io as io
import os

FRAME_COLOR = (255, 0, 0)
face_classifier = cv2.CascadeClassifier(
    'D:\Grad. Project\GP2\Codes\git_repo\Emotions_Detection\Assets\haarcascade_frontalface_default.xml')


def get_faces_from_image(image, color=FRAME_COLOR, is_dir=True, is_gray=True):
    '''
    Returns The coordinates of the detected faces in the image in a List object\n
    args:
        _img_: nd_array,
        color: Frame_color (depricated),
        is_dir: if passed img is a directory,
        is_gray: Boolean(True)
    '''

    img = io.imread(image) if is_dir else image
    global face_classifier
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.1, 5)

    return faces

if __name__ == '__main__':      
     print(os.getcwd())