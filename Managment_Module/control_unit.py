import random
import numpy as np
import pyttsx3

from speaker_managing import speaker_managment
# acceptable difference between the twwo areas of face to be considers same
AREA_THRESHOLD = 1

'''
##  Scheduling Module (Intelligent Control Unit):

    This module is responsible for managing when to say the next emotion and match each speaker with his emotion.
    
    Inputs: List<List<FaceData>> people
            | 1st person frame1     2nd person frame1     3rd person frame1    ..
            | 1st person frame2     2nd person frame2     3rd person frame2    ..
            |       :                       :                      :           :
            | 1st person frameN     2nd person frameN     3rd person frameN    ..
    
    Outputs: (Boolean to say text need cahnge or not , Text to be said and sent to text-to-speech module)
            i.e. (True, 'happy') or (False, 'same person')
    
'''


def test_cu(people, peopleNum):
    return (True, 'control unit called')

# helper functions:


def getPersonStatus(person, N):
    maskedFrames = 0
    mouthStates = []
    emotions = {}

    for frame in person:
        if frame.isMasked:
            maskedFrames += 1
        else:
            mouthStates.append(frame.mouthState)
            emotions[frame.emotion] = emotions.get(frame.emotion, 0) + 1
    # check if masked for almost of the frames
    if maskedFrames > N/2:
        return ('masked', None)

    # check if the person is speaking:
    speakingStatus = speaker_managment(person)
    emotion = max(emotions, key=emotions.get)

    return (speakingStatus, emotion)


def samePerson(person, prevPerson):
    return np.abs(person.face_area - prevPerson.face_area) < AREA_THRESHOLD

    # helper variables:
prevSpeakerNum = -1
prevPeople = None
prevPeopleStatus = None


def control_unit(people, peopleNum):
    # TODO complete the documentation and testing! testing -> test: scenarios , AREA_THRESHOLD ,
    # text is changed when:
    # 1. the emotion changes for the same speaker (e.g. happy -> sad)
    # 2. the speaker changed (new face position)

    # validation checks:
    if peopleNum == 0 or len(people) == 0:
        return (False, "No faces are detected or people have zero length")

    speakerNum = 0
    global prevSpeakerNum
    global prevPeople
    global prevPeopleStatus

    # preprocess people data:
    for i in range(peopleNum):
        people[i] = people[i][:peopleNum]
    people = np.array(people)
    people = np.transpose(people)
    prevPeople = people

    # TODO: do we have to check all people or the closest one only?
    # Ans: my answer is currently we only need the closest one but we might need all if we do further analysis

    # get the status of each person (speaking / not speaking / masked) with emotions:
    peopleStatus = []
    for person in people:
        status = getPersonStatus(person, len(person))
        if status[0] == 'Speaker':
            speakerNum += 1
        peopleStatus.append(status)
    prevPeopleStatus = peopleStatus

    # check if text needed to be changed:
    if prevSpeakerNum == speakerNum and speakerNum != 0:
        if samePerson(people[0], prevPeople[0]) and prevPeopleStatus[0][0] == peopleStatus[0][0]:
            return (False, "same person")
        else:
            # say emotion of closest one and speaking i.e. either new emotion for the same speaker
            # OR there is a new speaker
            return (True, peopleStatus[0][1])
    else:
        if speakerNum == 0 and peopleNum != 0:
            if peopleStatus[0][0] == 'masked':
                return (True, "masked")
            if samePerson(people[0], prevPeople[0]) and prevPeopleStatus[0][0] == peopleStatus[0][0]:
                return (False, "same person")
            else:  # say emotion of closest one and not speaking
                return (True, peopleStatus[0][1]+" but not speaking")
        elif speakerNum == 0 and peopleNum == 0:
            return (True, "No one around")
        elif prevSpeakerNum == 0:
            # and there must be a speaker (NEW) -> say his emotion
            return (True, peopleStatus[0][1])
        else:  # both prevSpeakerNum and speakerNum != 0 but different
            if samePerson(people[0], prevPeople[0]) and prevPeopleStatus[0][0] == peopleStatus[0][0]:
                return (False, "same person")
            else:  # say emotion of closest one and speaking
                return (True, peopleStatus[0][1])


if __name__ == '__main__':
    # start the text-to-speech engine:
    engine = pyttsx3.init()
    print('control unit main')
    engine.say('control unit main')
    engine.runAndWait()
