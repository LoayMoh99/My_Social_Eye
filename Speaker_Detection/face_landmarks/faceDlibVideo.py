import sys
import os
import dlib
import glob
import cv2
import time
import numpy as np

if len(sys.argv) != 2:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture("1.mpg")
success, img = cap.read()
prevTime = 0
features=np.zeros((25,1))
counter = 0
while success:

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        
        print(shape.part(61))

        upper1=np.asarray(shape.part(61))
        upper2=np.asarray(shape.part(62))
        upper3=np.asarray(shape.part(63))
        lower1=np.asarray(shape.part(67))
        lower2=np.asarray(shape.part(66))
        lower3=np.asarray(shape.part(65))
        
        print(type(upper1))
        print(lower1)

        #feat = np.linalg.norm(upper1-lower1)
        #print(feat)
        print("Upper: Part 61: {}, Part 62: {} ,Part 63 {} \nLower: Part 67: {}, Part 66: {} ,Part 65 {} ".format(shape.part(61),shape.part(62),shape.part(63),shape.part(67),shape.part(66),shape.part(65)))
                        

    success, img = cap.read()
    counter+=1
    
cap.release