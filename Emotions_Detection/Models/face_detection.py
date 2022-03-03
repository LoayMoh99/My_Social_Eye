import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

FRAME_COLOR = (255,0,0)

def get_faces_from_image(img, color=FRAME_COLOR, is_dir=True):
    '''
    Returns The coordinates of the detected faces in the image in a List opbject
    args:
        img: nd_array, color: Frame_color (depricated), is_dir: if passed img is a directory
    '''
    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    imgg = None
    if is_dir:
        imgg = io.imread(img)
    else: 
        imgg = img
    
    face_list = []
    gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in faces:
        # Draw Face Frame
        # cv2.rectangle(img,(x,y),(x+w,y+h), FRAME_COLOR, 2)
        # roi_gray = gray[y:y+h,x:x+w]
        face_list.append((x,y,w,h))
    
    return face_list
