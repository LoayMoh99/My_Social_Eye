
import threading
import cv2
from Speaker_Detection.mouth_detection.getMouthStateDlib import getMouthStateDlib
from Emotions_Detection.Models.face_detection import get_faces_from_image
from Emotions_Detection.main import EmotionDetectionModel
from Managment_Module.face_tracking_feature import extractFaceTrackFeature
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('./Managment_Module')
sys.path.append('./Speaker_Detection')
sys.path.append('./Emotions_Detection')
sys.path.append('./Speaker_Detection/mouth_detection')
sys.path.append('./Emotions_Detection/Models')


def getMouthState(frame, face):
    mouthState = getMouthStateDlib(frame, face)
    print('mouthState:', mouthState)
    return mouthState


def getEmotion(frame, face):
    # TODO : conjvert pred to pred_prob
    emotions = {'happy': 0.0, 'angry': 0.0,
                'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}
    emotionModel = EmotionDetectionModel(
        model_file='lpq_phog_model_old.sav', is_prob=False, feats='lpq_phog')
    emotionStr = emotionModel.get_labels(frame, face)
    #print('Detected Emotion:', emotionStr)
    emotions[emotionStr] = 1.0
    print('emotions:', emotions)
    return emotions
    # return getEmotionFER(frame, face)


def getFaceTrackFeature(frame, face):
    # get the face tracking feature (LBP)
    face_img = frame[face[1]:face[1] +
                     face[3], face[0]:face[0]+face[2]]
    faceTrack = extractFaceTrackFeature(face_img)
    return print('faceTrack:', faceTrack)


if __name__ == "__main__":
    frame = cv2.imread('./frame0.jpg')
    # detect faces in the image frame:
    faces = get_faces_from_image(frame, is_dir=False, is_gray=False)
    face = faces[0]

    # creating threads
    t1 = threading.Thread(target=getMouthState, name='t1', args=(frame, face))
    t2 = threading.Thread(target=getEmotion, name='t2', args=(frame, face))
    t3 = threading.Thread(target=getFaceTrackFeature,
                          name='t3', args=(frame, face))

    # starting threads
    t1.start()
    t2.start()
    t3.start()

    # wait until all threads finish
    t1.join()
    t2.join()
    t3.join()
