import glob
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
try:
    shutil.rmtree("./open_mouth/")
except:
    pass
try:
    shutil.rmtree("./close_mouth/")
except:
    pass
try:
    shutil.rmtree("./open_mouth_test/")
except:
    pass
try:
    shutil.rmtree("./close_mouth_test/")
except:
    pass

os.mkdir("./open_mouth/")
os.mkdir("./close_mouth/")
os.mkdir("./open_mouth_test/")
os.mkdir("./close_mouth_test/")

# constants:
DATASET_FOLDERS = 3
THRESHOLD_OPEN = 5.5
THRESHOLD_CLOSE = 1.5

for_test = True
for i in range(1, DATASET_FOLDERS+1):
    # : change the dataset source from here after extracting the zip folders to be s1 , s2 , ...
    DataSetDir = "C:\\Collage\\GP\\dataset\\s{i}\\".format(i=i)
    print(DataSetDir)
    for_test = True
    for filename in sorted(glob.glob(DataSetDir+'*.mpg')):
        cap = cv2.VideoCapture(filename)
        success, img = cap.read()
        all_count = 0
        open_count = 0
        close_count = 0
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

                if (feat > THRESHOLD_OPEN):
                    if for_test:
                        os.chdir("./open_mouth_test/")
                    else:
                        os.chdir("./open_mouth/")
                    cv2.imwrite(
                        "feat{feat} folder{i} - frame%d.jpg".format(feat=feat, i=i) % all_count, img)
                    os.chdir("../")  # go back to root directory
                    open_count += 1
                elif (feat < THRESHOLD_CLOSE):
                    if for_test:
                        os.chdir("./close_mouth_test/")
                    else:
                        os.chdir("./close_mouth/")
                    cv2.imwrite(
                        "feat{feat} folder{i} - frame%d.jpg".format(feat=feat, i=i) % all_count, img)
                    os.chdir("../")  # go back to root directory
                    close_count += 1
            success, img = cap.read()
            all_count += 1
        for_test = False
        print("For file ", filename, ":")
        print('Number of open mouth = ', open_count)
        print('Number of close mouth = ', close_count)
        print('Number of all frames = ', all_count)
        cap.release()
