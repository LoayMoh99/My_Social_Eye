
import numpy as np

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
    diff_bet_frames = 0
    # get the ratio of open:close frames (2nd metric)
    opened = 0
    mouthOpenNessDiscrete = np.zeros((len(mouthOpenNess), 1))
    for i in range(N):
        if mouthOpenNess[i] > 0.5:
            opened += 1
            mouthOpenNessDiscrete[i] = 1
        if i == 0:  # skip first value
            continue
        # diff_bet_frames += abs(mouthOpenNessDiscrete[i] -
        #                        mouthOpenNessDiscrete[i-1])
        diff_bet_frames += abs(mouthOpenNess[i] -
                               mouthOpenNess[i-1])
    ratio = opened / N

    # #2nd metric
    # if ratio > 0.25 and ratio < 0.75:
    #     return 'Speaker'
    # elif ratio <= 0.25:
    #     return 'Silent'
    # elif ratio >= 0.75:
    #     return 'Yawn'

    print(diff_bet_frames, N)
    # check if difference between open:close frames is more than 10% no. of frames (3rd metric)
    if diff_bet_frames > np.floor(0.1 * N):
        if ratio > 0.25 and ratio < 0.75:
            return 'Speaker'
        else:
            return 'Not Speaker'
    else:
        return 'Not Speaker'


if __name__ == '__main__':
    mouthOpenNessSpeaker = [0.7, 0.6, 0.2, 0.5, 0.6, 0.6, 0.5, 0.3]
    mouthOpenNessSlient = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    mouthOpenNessYawn = [0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    print(speaker_managment(mouthOpenNessSpeaker))
    print(speaker_managment(mouthOpenNessSlient))
    print(speaker_managment(mouthOpenNessYawn))
