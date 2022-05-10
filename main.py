
from Emotions_Detection.Models.face_detection import get_faces_from_image
from Emotions_Detection.fer_model import getEmotionFER
from Managment_Module.control_unit import test_cu
from Managment_Module.face_data_structure import FaceData
import cv2
import random
import sys

from Speaker_Detection.mouth_detection.getMouthStateDlib import getMouthStateDlib

sys.path.append('./Managment_Module')
sys.path.append('./Speaker_Detection')
sys.path.append('./Emotions_Detection')
sys.path.append('./Speaker_Detection/mouth_detection')
sys.path.append('./Emotions_Detection/Models')


# load the video:
TestDir = "C:\\Collage\\GP\\test\\"


def maskDetector(frame, face):
    return False


def getMouthState(frame, face):
    return getMouthStateDlib(frame, face)


def getEmotion(frame, face):
    return getEmotionFER(frame, face)


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


def main(isCamera=False, videoName=TestDir+"sleep.mp4"):
    # extract frames from video / camera
    cap = None
    if isCamera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videoName)
    success, frame = cap.read()

    F = 5  # frames per second
    S = 4  # seconds
    N = S * F  # number of frames

    def addToPeople(faceData):
        global people
        global peopleNum
        # add people count
        if peopleNum == -1:
            peopleNum = len(faceData)
        else:
            peopleNum = min(peopleNum, len(faceData))

        people.append(faceData)
        if len(people) == N:
            # call control unit
            decision = test_cu(people, peopleNum)
            if decision[0]:
                print("we will say the descision: " + decision[1])
            else:
                print("we will not say as " + decision[1])
            # remove first F frames
            people = people[F:]

    while success:
        # preprocess frame:
        #print("preprocessing frame if any")

        # detect faces in the image frame:
        faces = get_faces_from_image(frame, is_dir=False, is_gray=False)

        # create a list of FaceData objects:
        frameFaceData = []
        for face in faces:
            faceData = FaceData()

            # detect mask state for each face:
            faceData.isMasked = maskDetector(frame, face)

            if not faceData.isMasked:
                # detect mouth state for each face:
                faceData.mouthState = getMouthState(frame, face)
                print("Mouth Lips Distance:", faceData.mouthState)

                # detect emotions for each face:
                faceData.emotion = getEmotion(frame, face)
                print("Emotion:", faceData.emotion)

            # TODO: later make it sorted by face area (closer == first)
            frameFaceData.append(faceData)

        # update people list:
        addToPeople(frameFaceData)

        success, frame = cap.read()

    cap.release()


if __name__ == '__main__':
    print('Welcome to "My Social Eye"')

    main(isCamera=False, videoName=TestDir+"speaking.mp4")
