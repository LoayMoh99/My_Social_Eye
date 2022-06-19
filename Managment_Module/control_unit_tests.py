'''
This will be the main file for the control unit tests.

It will include all test cases for the control unit module.
'''

import unittest
import random
import numpy as np
import control_unit as cu
from face_data_structure import FaceData


class TestControlUnit(unittest.TestCase):
    F = 10
    N = 50
    people = []  # List<List<FaceData>>
    peopleNum = 2

    emotions = {'happy': 0.0, 'angry': 0.0,
                'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}
    # fill people with data
    for i in range(N):
        people.append([])
        for j in range(peopleNum):
            face = [random.randint(90, 100), random.randint(
                90, 100), random.randint(190, 200), random.randint(190, 200)]
            faceData = FaceData(face)
            faceData.isMasked = False
            emotions['happy'] = 1.0
            faceData.emotion = emotions
            faceData.faceTrackFeature = np.array(
                [47795.0, 25881.0, 20927.0,  5712.0, 24987.0, 17382.0,  4843.0, 23885.0, 21082.0, 42731.0])

            # 50% open and 50% close -> speaking for speaker1(j==0)
            if j == 0 and i % 2 == 0:
                faceData.mouthState = 1.0
            else:
                faceData.mouthState = 0.0

            people[i].append(faceData)

    def test_cu1(self):
        res = cu.test_cu(None, None)
        self.assertEqual(res, (True, 'control unit called'))

        # Detect speaker and happy
        res = cu.control_unit(people=self.people, peopleNum=self.peopleNum)
        print(res)
        self.assertEqual(res, (True, 'speaker is happy'))

        ###################################################################################
        # Detect that same speaker with same emotion -> Not saying anything
        people = self.people[2*self.F:]
        emotions = self.emotions
        for i in range(self.N-2*self.F, self.N):
            people.append([])
            for j in range(self.peopleNum):
                face = [random.randint(90, 100), random.randint(
                    90, 100), random.randint(190, 200), random.randint(190, 200)]
                faceData = FaceData(face)
                faceData.isMasked = False
                emotions['happy'] = 1.0
                faceData.emotion = emotions
                faceData.faceTrackFeature = np.array(
                    [47795.0, 25881.0, 20927.0,  5712.0, 24987.0, 17382.0,  4843.0, 23885.0, 21082.0, 42731.0])

                # 50% open and 50% close -> speaking for speaker1(j==0)
                if j == 0 and i % 2 == 0:
                    faceData.mouthState = 1.0
                else:
                    faceData.mouthState = 0.0

                people[i].append(faceData)

        res = cu.control_unit(people=people, peopleNum=self.peopleNum)
        print(res)
        self.assertEqual(res, (False, 'same speaker'))

        ###################################################################################
        # Detect that same speaker but with diff emotion -> say new emotion
        people = self.people[2*self.F:]
        emotions = self.emotions
        for i in range(self.N-2*self.F, self.N):
            people.append([])
            for j in range(self.peopleNum):
                face = [random.randint(90, 100), random.randint(
                    90, 100), random.randint(190, 200), random.randint(190, 200)]
                faceData = FaceData(face)
                faceData.isMasked = False
                emotions['happy'] = 0.0
                emotions['sad'] = 1.0
                faceData.emotion = emotions
                faceData.faceTrackFeature = np.array(
                    [47795.0, 25881.0, 20927.0,  5712.0, 24987.0, 17382.0,  4843.0, 23885.0, 21082.0, 42731.0])

                # 50% open and 50% close -> speaking for speaker1(j==0)
                if j == 0 and i % 2 == 0:
                    faceData.mouthState = 1.0
                else:
                    faceData.mouthState = 0.0

                people[i].append(faceData)

        res = cu.control_unit(people=people, peopleNum=self.peopleNum)
        print(res)
        self.assertEqual(res, (True, 'speaker is sad'))

    def test_cu2(self):
        ###################################################################################
        # Detect that diff speaker -> say the emotion of that new speaker
        people = []
        emotions = self.emotions
        for i in range(0, self.N):
            people.append([])
            for j in range(self.peopleNum):
                face = [random.randint(90, 100), random.randint(
                    90, 100), random.randint(190, 200), random.randint(190, 200)]
                faceData = FaceData(face)
                faceData.isMasked = False
                emotions['happy'] = 0.0
                emotions['sad'] = 1.0
                faceData.emotion = emotions
                # diff in there faceTrack feature
                faceData.faceTrackFeature = np.array(
                    [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0])

                # 50% open and 50% close -> speaking for speaker1(j==0)
                if j == 1 and i % 2 == 0:
                    faceData.mouthState = 1.0
                else:
                    faceData.mouthState = 0.0

                people[i].append(faceData)

        # Detect that same speaker but with diff emotion -> say new emotion
        res = cu.control_unit(people=people, peopleNum=self.peopleNum)
        print(res)
        self.assertEqual(res, (True, 'speaker is sad'))


if __name__ == '__main__':
    unittest.main()
