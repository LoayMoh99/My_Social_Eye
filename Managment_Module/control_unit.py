import random
import numpy as np
import pyttsx3

# acceptable difference between the twwo areas of face to be considers same
AREA_THRESHOLD = 30
XY_THRES = 30
TRACKING_THRES = 0.2


def test_cu(people, peopleNum):
    return (True, 'control unit called')

# helper functions:


def getPersonStatus(person, N):
    maskedFrames = 0
    mouthStates = []
    emotions = {}
    faceTrackingFeatures = []
    avgArea = 0
    avgPosition = [0, 0]
    for frame in person:
        # take average for position & area
        avgArea += frame.face_area
        avgPosition[0] += frame.face_position[0]
        avgPosition[1] += frame.face_position[1]
        faceTrackingFeatures.append(frame.faceTrackFeature)

        if frame.isMasked:
            maskedFrames += 1
        else:
            mouthStates.append(frame.mouthState)
            # emotions[frame.emotion] = emotions.get(frame.emotion, 0) + 1
            # add emotions to the dictionary
            for key, value in frame.emotion.items():
                if key in emotions:
                    emotions[key] += value
                else:
                    emotions[key] = value
    # check if masked for almost of the frames
    if maskedFrames > N/2:
        return ['masked', None, 0, [0, 0], None]

    # calculate average area and position
    avgArea /= N
    avgPosition[0] /= N
    avgPosition[1] /= N

    # calculate the average faceTrackingFeatures
    faceTrackingFeatures = np.array(faceTrackingFeatures)
    avgFaceTrackingFeature = np.mean(faceTrackingFeatures, axis=0)

    # check if the person is speaking:
    speakingStatus = speaker_managment(mouthStates)
    emotion = max(emotions, key=emotions.get)

    return [speakingStatus, emotion, avgArea, avgPosition, avgFaceTrackingFeature]


def samePerson(person, prevPerson):
    # make the check not only for the area ; it should compare the x,y cooardinates also
    # check if the face_tracking_feature didn't change much
    tracking_feat = np.linalg.norm(
        person[5]-prevPerson[5])*2 / (np.linalg.norm(person[5])+np.linalg.norm(prevPerson[5]))
    print('tracking_feat', tracking_feat)
    return (np.abs(person[4][0] - prevPerson[4][0]) < XY_THRES and
            np.abs(person[4][1] - prevPerson[4][1]) < XY_THRES and
            np.abs(person[3] - prevPerson[3]) < AREA_THRESHOLD and
            tracking_feat < TRACKING_THRES)


# helper variables:
prevSpeakerNum = -1
prevPeopleStatus = None


def control_unit(people, peopleNum):
    '''
    #  Control Unit:

        This module is responsible for managing when to say the next emotion and match each speaker with his emotion. \n

        Inputs: List<List<FaceData>> people
                | 1st person frame1     2nd person frame1     3rd person frame1    .. \n
                | 1st person frame2     2nd person frame2     3rd person frame2    .. \n
                |       :                       :                      :           :  \n
                | 1st person frameN     2nd person frameN     3rd person frameN    .. \n

        Outputs: (Boolean to say text need cahnge or not , Text to be said and sent to text-to-speech module) 
                i.e. (True, 'happy') or (False, 'same person')

    '''
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
    global prevPeopleStatus

    # preprocess people data:
    for i in range(len(people)):
        people[i] = people[i][:peopleNum]

    people = np.array(people).reshape(-1, peopleNum)
    people = np.transpose(people)

    # do we have to check all people or the closest one only?
    # Ans: my answer is currently we only need the closest one but we might need all if we do further analysis

    # get the status of each person (speaking / not speaking / masked) with emotions:
    peopleStatus = []
    for person in people:  # for each person
        status = getPersonStatus(person, len(person))
        # [speakingStatus, emotion, speakingValue, avgArea, avgPosition , avgFaceTrackingFeature]
        pepStatus = [status[0], status[1], 0, status[2], status[3], status[4]]
        if status[0] == 'Speaker':
            speakerNum += 1
            # 5000 is just a big number to give speaker a higher priority
            pepStatus[2] = 5000+status[2]
        elif status[0] == 'Not Speaker':  # not speaker
            pepStatus[2] = +status[2]
        else:
            pepStatus[2] = 0

        peopleStatus.append(pepStatus)
        peopleStatus = sorted(
            peopleStatus, key=lambda x: x[2], reverse=True)

    decision = None
    print('speakerNum , peopleNum: ', speakerNum, peopleNum)
    # check if text needed to be changed:
    if prevSpeakerNum == speakerNum and speakerNum != 0:
        if prevPeopleStatus != None and samePerson(prevPeopleStatus[0], peopleStatus[0]) and prevPeopleStatus[0][1] == peopleStatus[0][1]:
            decision = (False, "same speaker")
        else:
            # say emotion of closest one and speaking i.e. either new emotion for the same speaker
            # OR there is a new speaker
            decision = (True, peopleStatus[0][1])
    else:
        if speakerNum == 0 and peopleNum != 0:
            if peopleStatus[0][0] == 'masked':
                decision = (True, "masked")
            if prevPeopleStatus != None and prevSpeakerNum == 0 and samePerson(prevPeopleStatus[0], peopleStatus[0]) and prevPeopleStatus[0][1] == peopleStatus[0][1]:
                decision = (False, "same not speaker")
            else:  # say emotion of closest one and not speaking
                decision = (True, peopleStatus[0][1]+" but not speaking")
        elif speakerNum == 0 and peopleNum == 0:
            decision = (True, "No one around")
        elif prevSpeakerNum == 0:
            # and there must be a speaker (NEW) -> say his emotion
            decision = (True, peopleStatus[0][1])
        else:  # both prevSpeakerNum and speakerNum != 0 but different
            if prevPeopleStatus != None and prevSpeakerNum == 0 and samePerson(prevPeopleStatus[0], peopleStatus[0]) and prevPeopleStatus[0][1] == peopleStatus[0][1]:
                decision = (False, "same not speaker")
            else:  # say emotion of closest one and speaking
                decision = (True, peopleStatus[0][1])

    # update prevSpeakerNum and prevPeopleStatus:
    prevPeopleStatus = peopleStatus
    prevSpeakerNum = speakerNum

    return decision
#######################################################################################################
#######################################################################################################
#######################################################################################################


'''
# Speaker Managing Block:

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
    # # 2nd metric
    # if ratio > 0.25 and ratio < 0.75:
    #     return 'Speaker'
    # elif ratio <= 0.25:
    #     return 'Silent'
    # else:  # if ratio >= 0.75:
    #     return 'Yawn'
    #     (maxMouthOpenNess - minMouthOpenNess)

    diff_bet_frames = 0
    # get the ratio of open:close frames (2nd metric)
    opened = 0
    # mouthOpenNessDiscrete = np.zeros((len(mouthOpenNess), 1))
    for i in range(N):
        if mouthOpenNess[i] > 0.5:
            opened += 1
            # mouthOpenNessDiscrete[i] = 1
        if i == 0:  # skip first value
            continue
        # diff_bet_frames += abs(mouthOpenNessDiscrete[i] -
        #                        mouthOpenNessDiscrete[i-1])
        diff_bet_frames += abs(mouthOpenNess[i] -
                               mouthOpenNess[i-1])
    ratio = opened / N

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
