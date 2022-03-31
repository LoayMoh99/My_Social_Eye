import sys
import shutil
import os
import dlib
import cv2
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

# remove content open/close directories:
shutil.rmtree("./open_mouth/")
shutil.rmtree("./close_mouth/")

os.mkdir("./open_mouth/")
os.mkdir("./close_mouth/")


##################################
cap = cv2.VideoCapture("1.mpg")
success, img = cap.read()
count = 0
open = 0
close = 0
THRESHOLD_OPEN = 5.5
THRESHOLD_CLOSE = 2.5
while success:

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        # k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        upper1 = np.asarray([shape.part(61).x, shape.part(61).y])
        upper2 = np.asarray([shape.part(62).x, shape.part(62).y])
        upper3 = np.asarray([shape.part(63).x, shape.part(63).y])
        lower1 = np.asarray([shape.part(67).x, shape.part(67).y])
        lower2 = np.asarray([shape.part(66).x, shape.part(66).y])
        lower3 = np.asarray([shape.part(65).x, shape.part(65).y])

        feat = (np.linalg.norm(upper1-lower1) +
                np.linalg.norm(upper2-lower2)+np.linalg.norm(upper3-lower3))/3
        # print(feat)
        if (feat > THRESHOLD_OPEN):
            os.chdir("./open_mouth/")
            cv2.imwrite("frame%d.jpg" % count, img)
            os.chdir("../")  # go back to root directory
            print('open mouth')
            open += 1
        elif (feat < THRESHOLD_CLOSE):
            os.chdir("./close_mouth/")
            cv2.imwrite("frame%d.jpg" % count, img)
            os.chdir("../")  # go back to root directory
            print('close mouth')
            close += 1
    success, img = cap.read()
    count += 1

print('Number of open mouth = ', open)
print('Number of close mouth = ', close)
print('Number of all frames = ', count)
