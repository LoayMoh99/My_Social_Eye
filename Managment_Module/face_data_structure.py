class FaceData:
    def __init__(self, face_position=None, isMasked=None, emotion=None, mouthState=None):
        self.face_position = face_position
        self.face_area = face_position[3] if face_position!=None else 0  # this is the square root area
        self.isMasked = isMasked
        self.emotion = emotion
        self.mouthState = mouthState
