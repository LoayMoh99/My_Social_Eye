class FaceData:
    def __init__(self, face_position=None, isMasked=None, emotion=None, mouthState=None):
        self.face_position = face_position
        self.face_area = self.getFaceArea(face_position)
        self.isMasked = isMasked
        self.emotion = emotion
        self.mouthState = mouthState

    # TODO calculate the area from the face position
    def getFaceArea(self, face_position):
        area = 0
        return area
