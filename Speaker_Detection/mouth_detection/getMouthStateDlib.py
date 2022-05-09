
import numpy as np
import dlib


predictor_path = "C:/Collage/GP/My_Social_Eye/Speaker_Detection/face_landmarks/shape_predictor_68_face_landmarks.dat"

# load the dlib feature extractor:
predictor = dlib.shape_predictor(predictor_path)


def getMouthStateDlib(frame, face):
    # convert face to a dlib rectangle:
    dlib_rect = dlib.rectangle(
        left=face.left, top=face.top, right=face.right, bottom=face.bottom)
    shape = predictor(frame, dlib_rect)

    upper1 = np.asarray([shape.part(61).x, shape.part(61).y])
    upper2 = np.asarray([shape.part(62).x, shape.part(62).y])
    upper3 = np.asarray([shape.part(63).x, shape.part(63).y])
    lower1 = np.asarray([shape.part(67).x, shape.part(67).y])
    lower2 = np.asarray([shape.part(66).x, shape.part(66).y])
    lower3 = np.asarray([shape.part(65).x, shape.part(65).y])

    feat = (np.linalg.norm(upper1-lower1) +
            np.linalg.norm(upper2-lower2)+np.linalg.norm(upper3-lower3))/3

    return feat
