
import pyttsx3
from Speaker_Detection.mouth_detection.getMouthStateDlib import getMouthStateDlib
from Emotions_Detection.Models.face_detection import get_faces_from_image
from Emotions_Detection.main import EmotionDetectionModel
from Managment_Module.face_tracking_feature import extractFaceTrackFeature
from Managment_Module.control_unit import control_unit
from Managment_Module.control_unit import test_cu
from Managment_Module.face_data_structure import FaceData
import cv2
import sys
import warnings
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


def maskDetector(frame, face):
    return False


def getMouthState(frame, face):
    return getMouthStateDlib(frame, face)


def getEmotion(frame, face):
    # TODO : conjvert pred to pred_prob
    emotions = {'happy': 0.0, 'angry': 0.0,
                'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}
    emotionModel = EmotionDetectionModel(
        model_file='lpq_phog_model_old.sav', is_prob=False, feats='lpq_phog')
    emotionStr = emotionModel.get_labels(frame, face)
    #print('Detected Emotion:', emotionStr)
    emotions[emotionStr] = 1.0
    return emotions
    # return getEmotionFER(frame, face)


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


def main(isCamera=False, videoName=TestDir+"sleep.mp4"):
    # extract frames from video / camera
    cap = None
    if isCamera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videoName)
    success, frame = cap.read()

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
            print(decision)
            if decision[0]:
                print("we will say the descision: " + decision[1])
                text_to_speech(decision[1])
            else:
                print("we will not say as " + decision[1])
            # remove first F frames
            people = people[2*F:]
        elif numToSayNoFace == N:
            print("No faces are detected")
            numToSayNoFace = 0
    while cap.isOpened():
        if not success:
            # If loading a video, Use 'break instead of 'continue
            if isCamera:
                continue
            else:
                break
        # each model preprocess the frame as needed

        # detect faces in the image frame:
        faces = get_faces_from_image(frame, is_dir=False, is_gray=False)

        # create a list of FaceData objects:
        frameFaceData = []
        for face in faces:
            faceData = FaceData(face)

            # detect mask state for each face:
            faceData.isMasked = maskDetector(frame, face)

            # get the face tracking feature (LBP)
            face_img = frame[face[1]:face[1] +
                             face[3], face[0]:face[0]+face[2]]
            faceData.faceTrackFeature = extractFaceTrackFeature(
                face_img)

            if not faceData.isMasked:
                #: to be parallelized on different process/threads

                # detect mouth state for each face:
                faceData.mouthState = getMouthState(frame, face)
                #print("Mouth Lips Distance:", faceData.mouthState)

                # detect emotions for each face:
                faceData.emotion = getEmotion(frame, face)
                #print("Emotion:", faceData.emotion)

            # check if the face area is above a certain threshold
            if face[2] > APPROVED_AREA:
                frameFaceData.append(faceData)
                frameFaceData = sorted(
                    frameFaceData, key=lambda x: x.face_area, reverse=True)

        # update people list:
        addToPeople(frameFaceData)

        if isCamera:
            cv2.imshow('My Socail Eye', frame)
        success, frame = cap.read()
        # end with esc
        if cv2.waitKey(5) & 0XFF == 27:
            break
    # make sure everything is closed when exited
    cv2.destroyAllWindows()
    cap.release()


def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


if __name__ == '__main__':
    print('Welcome to "My Social Eye"')

    main(isCamera=True, videoName=TestDir+"test1.mp4")
