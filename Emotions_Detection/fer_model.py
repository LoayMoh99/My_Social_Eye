# Ready Made Model for emotion detection
from fer import FER
'''
first install fer:
    - pip install fer
'''


# a fast Emotion Detector to test CU:
emotion_detector = FER(mtcnn=True)


def getEmotionFER(frame, face) -> str:
    img = frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
    global emotion_detector
    emotion = emotion_detector.detect_emotions(img)[0]["emotions"]
    return emotion
