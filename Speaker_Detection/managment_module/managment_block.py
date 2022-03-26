import random
import numpy as np
import pyttsx3


'''
##  Managment Module:

    This module is responsible for managing when to say the next emotion and match each speaker with his emotion.
    
    Inputs: list of emotions, list of speechProb (speech / silence probability)
    
    Outputs: Text to be said and sent to text-to-speech module
    
'''


# helper variables:
prevSpeakerIndex = -1
prevSpeakerEmotion = ''


def managment_block(faceEmotions, speechProb):
    # text is changed when:
    # 1. the emotion changes for the same speaker (e.g. happy -> sad)
    # 2. the speaker changed (speechProb changes)

    # validation checks:
    if len(faceEmotions) == 0 or len(speechProb) == 0:
        return "No face detected ; faceEmotions or speechProb have zero length"
    if len(faceEmotions) != len(speechProb):
        return "Error: faceEmotions and speechProb have different lengths"

    global prevSpeakerIndex
    global prevSpeakerEmotion
    # get the max probability of speech (i.e. detect speaker index)
    speechProb = np.array(speechProb)
    speakerIndex = np.argmax(speechProb)
    if speakerIndex == prevSpeakerIndex:
        # if the speaker index is the same as the previous speaker index,
        # check if the emotion is changed or not
        if faceEmotions[speakerIndex] == prevSpeakerEmotion:
            # if the emotion is the same as the previous emotion,
            # don't change the text
            return 'No change !!'
        else:
            # the emotion is changed,
            # change the text
            prevSpeakerEmotion = faceEmotions[speakerIndex]
            return 'Emotion changed "{}" for the same speaker!'.format(faceEmotions[speakerIndex])
    else:
        # if the speaker index is different from the previous speaker index,
        # change the text
        prevSpeakerIndex = speakerIndex
        prevSpeakerEmotion = faceEmotions[speakerIndex]
        return 'Speaker changed with emotion "{}"'.format(faceEmotions[speakerIndex])


if __name__ == '__main__':
    # start the text-to-speech engine:
    engine = pyttsx3.init()
    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprised']

    # let's say that we detected 2 faces
    faceDetected = 2
    for i in range(6):
        # random testing:
        # faceEmotions = [emotions[random.randint(
        #     0, 4)] for i in range(faceDetected)]
        # speechProb = [random.random() for i in range(faceDetected)]
        faceEmotions = [emotions[0], emotions[1]]
        speechProb = [0.9, 0.1]
        # change the emotion for same speaker
        if i == 2:
            faceEmotions = [emotions[4], emotions[1]]
            speechProb = [0.9, 0.1]
        # change the speaker
        if i == 4:
            faceEmotions = [emotions[4], emotions[1]]
            speechProb = [0.2, 0.91]
        text = managment_block(faceEmotions, speechProb)
        print(text)
        engine.say(text)
        engine.runAndWait()
