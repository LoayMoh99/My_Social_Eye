
from Speaker_Detection.mouth_detection.speaker_managing import speaker_managment
from Speaker_Detection.feature_extractor import feature_extractor
import numpy as np
import cv2
import dlib
import pickle
import sys
sys.path.append('./Speaker_Detection')
sys.path.append('./Speaker_Detection/mouth_detection')


# load the face detector:
detector = dlib.get_frontal_face_detector()

# load the video:
TestDir = "C:\\Collage\\GP\\test\\"

# load our mouth open-ness detector model:
filename = 'Speaker_Detection\mouth_detection\dlib_model.sav'
mouthDetectorModel = pickle.load(open(filename, 'rb'))


def main(isCamera=False, videoName="sleep.mp4"):
    # extract frames from video / camera
    cap = None
    if isCamera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videoName)
    success, img = cap.read()
    mouthOpenNess = []
    feats = []
    while success:
        # detect faces in the image frame:
        dets = detector(img, 1)
        for _, d in enumerate(dets):
            feat = feature_extractor(img, d)
            feats.append(feat)

            # reshape to column vector
            sample_feature = np.array([feat]).reshape(-1, 1)
            sample_pred = mouthDetectorModel.predict_proba(sample_feature)
            mouthOpenNess.append(sample_pred[0][1])
            break
        success, img = cap.read()

    feats = np.array(feats)
    print(speaker_managment(mouthOpenNess))

    cap.release()


if __name__ == '__main__':
    print('Welcome to "My Social Eye"')

    main(isCamera=False, videoName=TestDir+"sleep.mp4")
