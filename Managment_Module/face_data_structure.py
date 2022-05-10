class FaceData:
    def __init__(self, face_position=[], isMasked=None, emotion=None, mouthState=None):
        self.face_position = face_position
        # this is the square root area
        self.face_area = face_position[3] if len(face_position) > 0 else 0
        self.isMasked = isMasked
        self.emotion = emotion
        self.mouthState = mouthState

    def __str__(self):
        return "FaceData: " + str(self.face_position) + " " + str(self.face_area)+" "+str(self.isMasked) + " " + str(self.emotion) + " " + str(self.mouthState)
