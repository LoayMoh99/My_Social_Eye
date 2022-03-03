from Models.face_detection import get_faces_from_image
from Models.LPQ import lpq
import cv2
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

# 1- Read the Image 
img = io.imread('tests/img6.jpg')
faces = get_faces_from_image(img, is_dir=False)
lpq_list = []
for (x,y,w,h) in faces:
    roi = img[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    print(roi.shape)
    # roi = roi[:,:,0]
    io.imshow(roi)
    io.show()
    res = lpq(roi,winSize=11, mode='h')
    print(res)
    lpq_list.append(res)

print(lpq_list[0].shape)
plt.bar(x=range(255),height=lpq_list[0])
plt.show()
  


