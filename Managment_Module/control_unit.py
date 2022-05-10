import random
import numpy as np
import pyttsx3

# acceptable difference between the twwo areas of face to be considers same
AREA_THRESHOLD = 30

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
            #### emotions[frame.emotion] = emotions.get(frame.emotion, 0) + 1
            # add emotions to the dictionary
            for key, value in frame.emotion.items():
                if key in emotions:
                    emotions[key] += value
                else:
                    emotions[key] = value
    # check if masked for almost of the frames
    if maskedFrames > N/2:
        return ('masked', None)

    # check if the person is speaking:
    speakingStatus = speaker_managment(mouthStates)
    emotion = max(emotions, key=emotions.get)

    return (speakingStatus, emotion)


def samePerson(person, prevPerson):
    # TODO make the check not only for the area ; it should compare the x,y cooardinates also
    return np.abs(person[0].face_area - prevPerson[0].face_area) < AREA_THRESHOLD

    # helper variables:
prevSpeakerNum = -1
prevPeople = None
prevPeopleStatus = None


def control_unit(people, peopleNum):
    # TODO complete the documentation and testing!
    # testing -> test: scenarios (v.v.imp -> will take some time), try AREA_THRESHOLD , ..

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
    for person in people:  # for each person
        status = getPersonStatus(person, len(person))
        print(status)
        if status[0] == 'Speaker':
            speakerNum += 1
        peopleStatus.append(status)
    prevPeopleStatus = peopleStatus
    prevSpeakerNum = speakerNum

    # check if text needed to be changed:
    if prevSpeakerNum == speakerNum and speakerNum != 0:
        if samePerson(people[0], prevPeople[0]) and prevPeopleStatus[0][0] == peopleStatus[0][0]:
            print("there is speaker but he is the same")
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

#######################################################################################################
#######################################################################################################
#######################################################################################################


'''
##  Speaker Managing Block:

    This block is responsible for managing if this series of mouth open-ness detected frames of a person is a speaker / silent / yawn(sleepy).
    
    Inputs: list of mouth opne-ness probability of (N frames , i.e N = 8) for specific person face.
                                    
    Outputs: This person is a speaker / not speaker {either: silent / yawn(sleepy)}
    
'''


def speaker_managment(mouthOpenNess) -> str:
    # speaking detection will be done by (1st metric):
    # - if the average of the mouth open-ness is greater than 0.25 and less than 0.75 -> speaker
    # - if the average of the mouth open-ness is less than 0.25 -> silent
    # - if the average of the mouth open-ness is greater than 0.75 -> yawn(sleepy)
    #           i.e as the average yawn time is almost ~ 5 sec
    # Another metric (2nd metric):
    # - if the ratio between open:all frames is between 0.25 and 0.75 -> speaker
    # - if the ratio between open:aLL frames is less than 0.25 and nearly equal -> silent
    # - if the ratio between open:aLL frames is greater than 0.75 and nearly equal -> yawn(sleepy)
    # Another metric (3rd metric):
    #   calculating the difference between all given frames mouth openness
    # - if diff. >  10% * num of frames gathered -> speaker
    # - else -> not speaker

    # validation checks:
    if len(mouthOpenNess) == 0:
        return "No face detected ; mouthOpenNess have zero length"

    # get the average of the mouth open-ness (1st metric)
    # avgMouthOpenNess = np.mean(mouthOpenNess)
    # if avgMouthOpenNess > 0.25 and avgMouthOpenNess < 0.75:
    #     return 'Speaker'
    # elif avgMouthOpenNess <= 0.25:
    #     return 'Silent'
    # elif avgMouthOpenNess >= 0.75:
    #     return 'Yawn'

    N = len(mouthOpenNess)
    # mouthOpenNess = np.array(mouthOpenNess)
    # minMouthOpenNess = np.min(mouthOpenNess)
    # maxMouthOpenNess = np.max(mouthOpenNess)
    # # min-max normalization
    # mouthOpenNess = (mouthOpenNess - minMouthOpenNess) / \
    #     (maxMouthOpenNess - minMouthOpenNess)

    diff_bet_frames = 0
    # get the ratio of open:close frames (2nd metric)
    opened = 0
    #mouthOpenNessDiscrete = np.zeros((len(mouthOpenNess), 1))
    for i in range(N):
        if mouthOpenNess[i] > 0.5:
            opened += 1
            #mouthOpenNessDiscrete[i] = 1
        if i == 0:  # skip first value
            continue
        # diff_bet_frames += abs(mouthOpenNessDiscrete[i] -
        #                        mouthOpenNessDiscrete[i-1])
        diff_bet_frames += abs(mouthOpenNess[i] -
                               mouthOpenNess[i-1])
    ratio = opened / N

    # # 2nd metric
    # if ratio > 0.25 and ratio < 0.75:
    #     return 'Speaker'
    # elif ratio <= 0.25:
    #     return 'Silent'
    # else:  # if ratio >= 0.75:
    #     return 'Yawn'

    print(diff_bet_frames, ratio, N)
    # check if difference between open:close frames is more than 10% no. of frames (3rd metric)
    if diff_bet_frames > np.floor(0.1 * N):
        if ratio > 0.25 and ratio < 0.75:
            return 'Speaker'
        else:
            return 'Not Speaker'
    else:
        return 'Not Speaker'


#######################################################################################################
#######################################################################################################
#######################################################################################################

if __name__ == '__main__':
    # start the text-to-speech engine:
    engine = pyttsx3.init()
    print('control unit main')
    engine.say('control unit main')
    engine.runAndWait()
