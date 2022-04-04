import sys
import os
import dlib
import glob
import cv2
import time

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
win = dlib.image_window()
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("1.mpg")
prevTime = 0
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, Use 'break instead of 'continue
        continue
    
    #img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

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
        print("Upper: Part 61: {}, Part 62: {} ,Part 63 {} \nLower: Part 67: {}, Part 66: {} ,Part 65 {} ".format(shape.part(61),shape.part(62),shape.part(63),shape.part(67),shape.part(66),shape.part(65)))
              
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    win.add_overlay(dets)
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    print("fps= ",fps)
    dlib.hit_enter_to_continue
    
cap.release
