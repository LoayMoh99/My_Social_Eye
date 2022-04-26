import cv2
import glob

class DatasetReader():

    @staticmethod
    def read_dataset(directory):
        #Images list
        x_train = []
        y_train = []
        #(anger, disgust, fear, happiness, neutral, sadness and surprise)
        #(  0  ,    1   ,  2  ,     3    ,    4   ,    5    and     6   )
        for image in glob.glob(directory+ "*.jpg"):
            image = image.replace("\\", "/")
            image_name = image.split("/")[-1]
          
            expressionID = image_name.split("_")[0]
            x_train.append(cv2.imread(image))
            y_train.append(int(expressionID))
        
        return x_train, y_train


