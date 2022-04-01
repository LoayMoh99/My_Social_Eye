# This is for testing the speaking detection block integrated:
import pickle
import numpy as np
import cv2
import dlib
import speaker_managing as spk_mng


predictor_path = "C:/Collage/GP/My_Social_Eye/speaker_detection/face_landmarks/shape_predictor_68_face_landmarks.dat"

# load the dlib feature extractor and face detector:
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# load our mouth open-ness detector model:
filename = 'dlib_model.sav'
mouthDetectorModel = pickle.load(open(filename, 'rb'))

# load the video:
TestDir = "C:\\Collage\\GP\\test\\"


def integrated_speaaker_detector(videoName="sleep"):
    cap = cv2.VideoCapture(TestDir+videoName+".mp4")
    success, img = cap.read()
    frame_num = 0
    mouthOpenNess = []
    feats = []
    while success:
        dets = detector(img, 1)
        for _, d in enumerate(dets):
            shape = predictor(img, d)

            upper1 = np.asarray([shape.part(61).x, shape.part(61).y])
            upper2 = np.asarray([shape.part(62).x, shape.part(62).y])
            upper3 = np.asarray([shape.part(63).x, shape.part(63).y])
            lower1 = np.asarray([shape.part(67).x, shape.part(67).y])
            lower2 = np.asarray([shape.part(66).x, shape.part(66).y])
            lower3 = np.asarray([shape.part(65).x, shape.part(65).y])

            feat = (np.linalg.norm(upper1-lower1) +
                    np.linalg.norm(upper2-lower2)+np.linalg.norm(upper3-lower3))/3
            # print(
            #     'For frame:{frame_num} - feat={feat} '.format(frame_num=frame_num, feat=feat))
            feats.append(feat)

            # reshape to column vector
            sample_feature = np.array([feat]).reshape(-1, 1)
            sample_pred = mouthDetectorModel.predict_proba(sample_feature)
            mouthOpenNess.append(sample_pred[0][1])
            break
        frame_num += 1
        success, img = cap.read()
    feats = np.array(feats)
    print('Avg. feat={avg} '.format(avg=np.mean(feats)))

    cap.release()

    print(spk_mng.speaker_managment(mouthOpenNess))


if __name__ == '__main__':
    # Expected: silent (no speaking)
    integrated_speaaker_detector("no_speak")

    # Expected: sleep (Yawn)
    integrated_speaaker_detector("sleep")

    # Expected: speaking
    integrated_speaaker_detector("speaking")

    # Expected: speaking also but with low lips motion "Challenging"
    integrated_speaaker_detector("speaking_challenge")
