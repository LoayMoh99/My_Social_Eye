import os
import numpy as np
import cv2

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

class EmotionsDetectionModel:
    def __init__(self, verbose: bool=False) -> None:
        self.verbose = verbose
        self.model = self._create_model()
        weight_path = os.path.join(os.getcwd(), 'Emotions_Detection/Models/model_pretrained.h5')
        self.model.load_weights(weight_path)
        self.emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    def predict_image(self, frame: np.array, rect=None) -> str:
        '''Receives an image of any size with 2 or 3 color channels max'''
        # 1- Preprocess the Image
        x, y, w, h = rect
        face = frame[y:y+h,x:x+w]
        # Convert Image to Gray
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) >= 3 else face
        face = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)

        # Recognize the Emotion -> Predict
        pred = self.model.predict([face], verbose=self.verbose)
        emotion = self.emotions[np.argmax(pred)]
        return emotion if emotion not in ['Sad', 'Angry', 'Fearful'] else 'Unpleased'

    def _create_model(self) -> Sequential:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        return model