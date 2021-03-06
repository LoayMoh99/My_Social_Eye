
import numpy as np
import dlib
import os 

# get curr dir
dir = os.getcwd()
print(dir)
# load the dlib feature extractor:
predictor_path = dir+"/face_landmarks/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


def mouth_feature_extractor(img, face):
    global predictor
    shape = predictor(img, face)

    upper1 = np.asarray([shape.part(61).x, shape.part(61).y])
    upper2 = np.asarray([shape.part(62).x, shape.part(62).y])
    upper3 = np.asarray([shape.part(63).x, shape.part(63).y])
    lower1 = np.asarray([shape.part(67).x, shape.part(67).y])
    lower2 = np.asarray([shape.part(66).x, shape.part(66).y])
    lower3 = np.asarray([shape.part(65).x, shape.part(65).y])

    feat = (np.linalg.norm(upper1-lower1) +
            np.linalg.norm(upper2-lower2)+np.linalg.norm(upper3-lower3))/3

    return feat
