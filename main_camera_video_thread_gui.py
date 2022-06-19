
import threading
import pyttsx3
from Speaker_Detection.mouth_detection.getMouthStateDlib import getMouthStateDlib
from Emotions_Detection.Models.face_detection import get_faces_from_image
from Emotions_Detection.main import EmotionDetectionModel
from Managment_Module.face_tracking_feature import extractFaceTrackFeature
from Managment_Module.control_unit import control_unit
from Managment_Module.control_unit import test_cu
from Managment_Module.face_data_structure import FaceData
import cv2
import numpy as np
import sys
import warnings
# for gui
from tkinter import *
from PIL import ImageTk, Image
import cv2
import keyboard
warnings.filterwarnings('ignore')

sys.path.append('./Managment_Module')
sys.path.append('./Speaker_Detection')
sys.path.append('./Emotions_Detection')
sys.path.append('./Speaker_Detection/mouth_detection')
sys.path.append('./Emotions_Detection/Models')
# start the text-to-speech engine:
engine = pyttsx3.init()

# load the video:
TestDir = "C:\\Collage\\GP\\test\\"

# Global Frame Variables for Threading
globalEmotion = None
globalMouthState = None
globalFaceTrack = None


def maskDetector(frame, face):
    return False


def getMouthState(frame, face):
    global globalMouthState
    globalMouthState = getMouthStateDlib(frame, face)
    # return getMouthStateDlib(frame, face)


def getEmotion(frame, face):
    # conjvert pred to pred_prob
    '''
    emotionModel = EmotionDetectionModel(
        model_file='lpq_prob.sav', is_prob=True, feats='lpq_phog')
    emotions = emotionModel.get_labels_prob(frame, face)
    '''
    emotions = {'angry': 0.0, 'happy': 0.0, 'neutral': 0.0,
                'sad': 0.0, 'surprised': 0.0}
    # emotionModel = EmotionDetectionModel(
    #     model_file='lpq_phog_model.sav', is_prob=True, feats='lpq_phog')
    # emotions_arr = emotionModel.get_labels(frame, face)
    # emotions['anger'] = emotions_arr[0]
    # emotions['happy'] = emotions_arr[1]
    # emotions['neutral'] = emotions_arr[2]
    # emotions['sad'] = emotions_arr[3]
    # emotions['surprise'] = emotions_arr[4]
    emotionModel = EmotionDetectionModel(
        model_file='lpq_phog_model_old.sav', is_prob=False, feats='lpq_phog')
    emotionStr = emotionModel.get_labels(frame, face)
    emotions[emotionStr] = 1.0
    print(emotions)
    global globalEmotion
    globalEmotion = emotions
    # return emotions
    # return getEmotionFER(frame, face)


def getFaceTrackFeature(frame, face):
    # get the face tracking feature (LBP)
    face_img = frame[face[1]:face[1] +
                     face[3], face[0]:face[0]+face[2]]
    global globalFaceTrack
    globalFaceTrack = extractFaceTrackFeature(face_img)


'''
    Integration Algorithm:
        -   say we take F frames per second ; F = 5
        -   for S seconds ; S = 4 -> we have N = S*F = 20 frames
        -   for each frame: detect faces (no. of people) 
            *   for each face: check if masked 
                *   if not masked: check emotion / mouth_state
        -   for N=20 frames: 
            get actual no. of people {{ consider only faces that on all frames & are detected }}
            *   loop on all persons (by no. of people):
                *   if masked for almost all frames  -> "masked"
                *   else:
                    *   detect if speaking
                    *   detect his average emotions for this N frames
                Now we have no. of speaking people, no. of people and people matrix <FaceData>
                i.e. sample output at this point:
                noPeople = 3 , noSpeakingPeople = 1, people = [(speaking,happy), (not_speaking,neutral), (masked,-1)]
        -   the output will be entered to the control unit to detect whether to say a new thing or not
            it works cyclic; means that the N frames is updated every second (every F frames)

'''

# this will contain the data for N frames
people = []
peopleNum = -1
numToSayNoFace = 0
APPROVED_AREA = 50


def main_camera_thread_gui(isCamera=False, camNum=0, videoName=TestDir+"sleep.mp4", silent=False, root=None, label=None, logo=None, ins=None):
    # extract frames from video / camera
    cap = None
    if isCamera:
        cap = cv2.VideoCapture(camNum)
    else:
        cap = cv2.VideoCapture(videoName)

    results = []
    F = 10  # frames per second
    S = 5  # seconds
    N = S * F  # number of frames

    def addToPeople(faceDataList):
        global people
        global peopleNum
        global numToSayNoFace
        # handle this case properly as num sometimes is 0
        # else:
        #     peopleNum = min(peopleNum, len(faceDataList))

        # add people count
        if len(faceDataList) > 0:
            if peopleNum == -1:
                peopleNum = len(faceDataList)
            else:
                peopleNum = min(peopleNum, len(faceDataList))

            people.append(faceDataList)
        else:
            numToSayNoFace += 1

        if len(people) == N:
            # call control unit
            #decision = test_cu(people, peopleNum)
            decision = control_unit(people, peopleNum)
            results.append(decision)
            print(decision)
            if decision[0]:
                ins.config(text=decision[1])
                print("we will say the descision: " + decision[1])
                if not silent:
                    text_to_speech(decision[1])

            else:
                print("we will not say as " + decision[1])
                ins.config(text="we will not say as " + decision[1])
            root.update()
            # remove first F frames
            people = people[2*F:]
        elif numToSayNoFace == N:
            print("No faces are detected")
            ins.config(text="No faces are detected")
            numToSayNoFace = 0

    while True:
        frame = cap.read()[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # each model preprocess the frame as needed

        # detect faces in the image frame:
        faces = get_faces_from_image(frame, is_dir=False, is_gray=False)

        # create a list of FaceData objects:
        frameFaceData = []
        for face in faces:
            faceData = FaceData(face)

            # detect mask state for each face:
            faceData.isMasked = maskDetector(frame, face)

            if not faceData.isMasked:
                #: to be parallelized on different process/threads
                ########################################
                # creating threads
                t1 = threading.Thread(target=getMouthState,
                                      name='t1', args=(frame, face))
                t2 = threading.Thread(
                    target=getEmotion, name='t2', args=(frame, face))
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
                ########################################
                # detect mouth state for each face:
                global globalMouthState
                faceData.mouthState = globalMouthState
                #print("Mouth Lips Distance:", faceData.mouthState)

                # detect emotions for each face:
                global globalEmotion
                faceData.emotion = globalEmotion
                #print("Emotion:", faceData.emotion)

                # detect face tracking feature for each face:
                global globalFaceTrack
                faceData.faceTrackFeature = globalFaceTrack

            # check if the face area is above a certain threshold
            if face[2] > APPROVED_AREA:
                frameFaceData.append(faceData)
                frameFaceData = sorted(
                    frameFaceData, key=lambda x: x.face_area, reverse=True)

        # update people list:
        addToPeople(frameFaceData)

        # show the frame
        frame = ImageTk.PhotoImage(Image.fromarray(frame))
        label.config(image=frame)
        root.update()
        # Check if S Button was pressed
        if keyboard.is_pressed('s'):
            label.config(image=logo)
            break


def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()
