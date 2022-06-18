import time
from matplotlib.pyplot import show
from commonfunctions import *
import math
from skimage import filters
from skimage import data
from skimage.color import rgb2gray

import skfuzzy as fuzz
import dlib



def detectMouth(img):
    # detect face:
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    # print("Number of faces detected: {}".format(len(dets)))

    d=dets[0]

    face = img[d.top():d.bottom(), d.left():d.right()]
    dimensions_face=face.shape
    mouth = face[int(dimensions_face[0]/2+(0.05*(d.bottom()-d.top()))):,:,:]


    quarter_face=mouth.shape[1]//4
    mouth= mouth[:,quarter_face:quarter_face*3,:]

    dimensions_mouth=mouth.shape

    # io.imshow(face)
    # io.imshow(mouth)
    # io.show()
    return mouth , dimensions_mouth


def PixelsCount(image):
    return image.shape[0]*image.shape[1]


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def selectLipRegion(mouth, dimensions_mouth):
    R = mouth[:,:,0]
    G = mouth[:,:,1]

    # calculate chromatism
    chromatism = 2 * np.arctan((R-G)/R) / np.pi

    chromatism = chromatism.reshape(-1,1)  # reshape to a vector

    sorted_chromatism = -np.sort(-chromatism,axis=None,)    # sort descendingly 

    chromatism_mouth_perc = int(chromatism.shape[0] * 0.15) 
    chromatsim_largest_twenty=sorted_chromatism[0:chromatism_mouth_perc]  #


    chromatism = chromatism.reshape(dimensions_mouth[0],dimensions_mouth[1])
    # io.imshow(chromatism)
    # io.show()

    new_img = np.zeros(dimensions_mouth)

    for i in range(0,dimensions_mouth[0]):
        for j in range(0,dimensions_mouth[1]):
            if chromatism[i,j] in chromatsim_largest_twenty:
                new_img[i,j]=255
            else:
                new_img[i,j]=0

    return new_img

def contourIntersect(contour1, contour2):
    intersection = cv2.pointPolygonTest(contour1,(int(contour2[0][0][0]),int(contour2[0][0][1])),False)
    return intersection


def getROI(mouth,mouth_original,contours):

    #showing contours
    temp = np.zeros_like(mouth)
    #one connected component or 2 connected component
    connected = True  

    # get 2 largest contours 
    largest_two_cnts=contours[0:2]  
    cv2.drawContours(temp, largest_two_cnts, -1, (255,0,0), 1) #draw all contours


    #if 2 components are nested , inside = 1 , else , components are not connected
    inside = contourIntersect(largest_two_cnts[0],largest_two_cnts[1])   

    if inside>0:
        connected=True
    else:
        connected=False

    if connected: # if connected , draw bounding rect for largest (outer contour)
        x,y,w,h = cv2.boundingRect(largest_two_cnts[0])
        #cv2.rectangle(temp,(x,y),(x+w,y+h),(255,255,255),1)
        # print("connected ",x,y,w,h)
        ROI = mouth_original[max(2,y-int(0.10*mouth.shape[0])):min(int(y+h+int(0.10*mouth.shape[0])),mouth.shape[0]),max(2,x-int(0.2*mouth.shape[1])):min(mouth.shape[1],x+w+int(0.2*mouth.shape[1])),:]
    else:   #else , draw bounding rect for both region
        x1,y1,w1,h1 = cv2.boundingRect(largest_two_cnts[0])     
        #cv2.rectangle(temp,(x1,y1),(x1+w1,y1+h1),(255,255,255),1)

        x2,y2,w2,h2 = cv2.boundingRect(largest_two_cnts[1])
        #cv2.rectangle(temp,(x2,y2),(x2+w2,y2+h2),(255,255,255),1)

        # print("area difference : ",cv2.contourArea(largest_two_cnts[0])-cv2.contourArea(largest_two_cnts[1]))           
        if cv2.contourArea(largest_two_cnts[0])-cv2.contourArea(largest_two_cnts[1]) >500:      #if the size of second largest is negligible in comparison to first , discard it
            # print("not connected ",x1,y1,w1,h1)
            ROI = mouth_original[max(2,y1-int(0.10*mouth.shape[0])):min(mouth.shape[0],y1+h1+int(0.10*mouth.shape[0])),max(2,x1-int(0.2*mouth.shape[1])):min(mouth.shape[1],x1+w1+int(0.2*mouth.shape[1])),:]
        else:
            minx=min([x1,x2])               #get the minx maxx miny maxy of the 2 bounding rectangles combined
            maxx=max([x1+w1,x2+w2])
            miny=min([y1,y2])
            maxy=max([y1+h1,y2+h2])
            # print("not connected ",miny,maxy,minx,maxx)
            ROI = mouth_original[max(2,miny-int(0.10*mouth.shape[0])):min(mouth.shape[0],maxy+int(0.10*mouth.shape[0])),max(2,minx-int(0.2*mouth.shape[1])):min(mouth.shape[1],maxx+int(0.2*mouth.shape[1])),:]         #retrieve ROI

    return ROI



def getExternalContour_Ycbcr(ROI):

    # print(ROI.shape)
    ROI_Ycbcr = rgb2ycbcr(ROI)

    Y = ROI_Ycbcr[:,:,0]
    cb = ROI_Ycbcr[:,:,1]
    cr = ROI_Ycbcr[:,:,2]
    
    N1=0
    S1=0
    nk=dict()
    for i in range(cb.shape[0]):
        for j in range(cb.shape[1]):
            for k in range(105,256):
                if cb[i,j] == k and (k not in nk):
                    nk[k] = 1
                elif  cb[i,j] == k:
                    nk[k]+=1

    # print(nk)
        
    # print(nk)

    #lip color and color of skin of Cb component are more than 100
    #ni : number of pixels who’s value is i
    #N : the number of pixels who’s grey value is greater than a certain gray level

    N1 = sum(nk.values())
    if N1 > 0:

        S1 = sum([key * value for key,value in nk.items()])

        t1 = S1 / N1
    else:
        t1 = 130

    #lip color and color of skin of Cb component are more than 100
    #ni : number of pixels who’s value is i
    #N : the number of pixels who’s grey value is greater than a certain gray level

    N2=0
    S2=0
    nk=dict()
    for i in range(cb.shape[0]):
        for j in range(cb.shape[1]):
            for k in range(145,256):
                if cb[i,j] == k and (k not in nk):
                    nk[k] = 1
                elif  cb[i,j] == k:
                    nk[k]+=1
    
    # print(nk)
    

    N2 = sum(nk.values())
    if N2 > 0:
        S2 = sum([key * value for key,value in nk.items()])

        t2 = S2 / N2
    
    else:
        t2 = 150

    # print(nk)

    thresh1 = int((t1 + t2)/2)
    thresh2 = int(t2)

    cbcr_avg = np.uint8((np.int16(cb)+np.int16(cr))/2)
    cbcr_avg_binary = cbcr_avg > thresh1
    cr_binary = cr > thresh2
    lip_noisy = np.logical_and(cbcr_avg_binary,cr_binary)
    lip_cleaned = np.uint8(lip_noisy*255) # convert the clean image into uint8

    return lip_cleaned


def getFinalRatio(lip_cleaned):
    contours, hierarchy = cv2.findContours(lip_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # get cnts of the cleaned image
    contours=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True) #sorted contours descendingly according to area
    largest_cnts=contours[0:2] # largest cnt
    # final_img = np.zeros((lip_cleaned.shape[0],lip_cleaned.shape[1],3)) # create 

    inside = contourIntersect(largest_cnts[0],largest_cnts[1]) 

    # print(inside)
    if inside>0:
        connected=True
    else:
        connected=False

    if connected:
        x1,y1,w1,h1 = cv2.boundingRect(largest_cnts[0])     
        #cv2.rectangle(lip_cleaned,(x1,y1),(x1+w1,y1+h1),(255,255,255),1)
        final_width = w1
        final_height = h1
    else:
        x1,y1,w1,h1 = cv2.boundingRect(largest_cnts[0])     
        #cv2.rectangle(lip_cleaned,(x1,y1),(x1+w1,y1+h1),(255,255,255),1)

        x2,y2,w2,h2 = cv2.boundingRect(largest_cnts[1])
        #cv2.rectangle(lip_cleaned,(x2,y2),(x2+w2,y2+h2),(255,255,255),1)

        minx=min([x1,x2])               #get the minx maxx miny maxy of the 2 bounding rectangles combined
        maxx=max([x1+w1,x2+w2])
        miny=min([y1,y2])
        maxy=max([y1+h1,y2+h2])
        final_width = maxx-minx
        final_height = maxy-miny


    final_feature =  (final_height)/(final_width)
    print("height / width = : ", final_feature )


def ReadImage(path):
    
    img = io.imread(path)
    img = (resize(img,(480,360,3),anti_aliasing=True)*255).astype("uint8")
    return img




#################################

def getFinalRegionOfInterest(img):
    mouth_original , dim_original = detectMouth(img)
    mouth = selectLipRegion(mouth_original, dim_original)
    mouth = np.uint8( rgb2gray(mouth) *255 ) 


    contours, hierarchy = cv2.findContours(mouth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ####################################

    contours=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True) #sorted contours descendingly according to area

    #############################

    ROI = getROI(mouth,mouth_original,contours)
    return ROI



#main function
if __name__ == "__main__":

    start = time.time()

    img = ReadImage('./test_images/00513_940519_fa_a.ppm')
    ROI  = getFinalRegionOfInterest(img)
    lip_cleaned = getExternalContour_Ycbcr(ROI)
    getFinalRatio(lip_cleaned)

    end = time.time()

    elapsed = end - start
    print("time elapsed for one frame in seconds " ,elapsed)






