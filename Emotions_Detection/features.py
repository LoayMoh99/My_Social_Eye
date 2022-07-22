import numpy as np
import cv2
from scipy.signal import convolve2d
import dask
from dask import delayed

#############################################################################################################################
class LPQ:
    def __init__(self, win_size=11, alpha=1/11) -> None:
        WIN_SIZE: int = win_size
        ALPHA: float = alpha
        WIN_RADIUS: int = (WIN_SIZE - 1) // 2 

        self.x = np.arange(-WIN_RADIUS,WIN_RADIUS+1)[np.newaxis]
        self.w0 = np.ones_like(self.x)
        self.w1 = np.exp(-2*np.pi*self.x*ALPHA*1j)
        self.w2 = np.conj(self.w1)

    def compute(self, image):
        filterResp1 = convolve2d(convolve2d(image, self.w0.T, 'valid'),self.w1,'valid')
        filterResp2 = convolve2d(convolve2d(image, self.w1.T, 'valid'),self.w0,'valid')
        filterResp3 = convolve2d(convolve2d(image, self.w1.T, 'valid'),self.w1,'valid')
        filterResp4 = convolve2d(convolve2d(image, self.w1.T, 'valid'),self.w2,'valid')

        # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
        freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

        ## Perform quantization and compute LPQ codewords
        inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
        LPQdesc=((freqResp>0)*(2**inds)).sum(2)

        LPQdesc = np.histogram(LPQdesc.flatten(),range(256))[0]
        return LPQdesc / LPQdesc.max()
    
 #############################################################################################################################   
class LPQPlus:
    def __init__(self, win_size=11, q_levels=8) -> None:
        WIN_SIZE: int = win_size
        ALPHA: float = 1 / WIN_SIZE
        CONV_MODE:str = 'valid'
        WIN_RADIUS: int = (WIN_SIZE - 1) // 2 
        Q_LEVELS: int = q_levels
        FUND_ANGLE: float = (2 * np.pi) / Q_LEVELS
        self.VOTE_ALPHA: int = 2

        self.x = np.arange(-WIN_RADIUS,WIN_RADIUS+1)[np.newaxis]
        self.w0 = np.ones_like(self.x)
        self.w1 = np.exp(-2*np.pi*self.x*ALPHA*1j)
        self.w2 = np.conj(self.w1)

        self.ANGLES = [FUND_ANGLE * i for i in range(Q_LEVELS)]

    def compute(self, image):
        print(image.shape)
        # Applying filters and getting response
        filterResp1: np.ndarray = convolve2d(convolve2d(image, self.w0.T, 'valid'),self.w1,'valid') # (0, a)
        filterResp2: np.ndarray = convolve2d(convolve2d(image, self.w1.T, 'valid'),self.w0,'valid') # (a, 0)
        filterResp3: np.ndarray = convolve2d(convolve2d(image, self.w1.T, 'valid'),self.w1,'valid') # (a, a)
        filterResp4: np.ndarray = convolve2d(convolve2d(image, self.w1.T, 'valid'),self.w2,'valid') # (a, -a)

        # Local Phases of the Responses
        theta1 = np.arctan2(filterResp1.imag, filterResp1.real)
        theta2 = np.arctan2(filterResp2.imag, filterResp2.real)
        theta3 = np.arctan2(filterResp3.imag, filterResp3.real)
        theta4 = np.arctan2(filterResp4.imag, filterResp4.real)

        vote1 = np.array(list(map(lambda theta: (np.cos(theta1 - theta)**self.VOTE_ALPHA).sum(), self.ANGLES)))
        vote2 = np.array(list(map(lambda theta: (np.cos(theta2 - theta)**self.VOTE_ALPHA).sum(), self.ANGLES)))
        vote3 = np.array(list(map(lambda theta: (np.cos(theta3 - theta)**self.VOTE_ALPHA).sum(), self.ANGLES)))
        vote4 = np.array(list(map(lambda theta: (np.cos(theta4 - theta)**self.VOTE_ALPHA).sum(), self.ANGLES)))
        lpq_desc = np.concatenate((vote1, vote2, vote3, vote4))

        # Normailze the data
        return  lpq_desc / lpq_desc.max()
        
#############################################################################################################################
    def compute_prll(self, imgs):
        # 1- set Function as delayed
        # 2- for loop for all of the images
        #   2.1  Apply LPQ(+) for every img
        descs = []
        for img in imgs:
            desc = delayed(self.compute)(img)
            descs.append(desc)

        return dask.compute(descs)[0]
    
#############################################################################################################################
def hogFunc(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    print(hog)
    return h
#############################################################################################################################
def PHOG_Algorithm(image,numberOfBins=8,numberOfLevels=3):#Reading Image as gray scale
    dummy_value=1e-5
    degree=360/numberOfBins
    x,y=image.shape
    pyramidarr=np.array([])
    #Intializing binary histogram array and binary gradient values array 
    binaryHist = np.zeros((x,y))
    binaryGrad = np.zeros((x,y))
    
    if np.sum(image) >100:
        medianImg= np.median(image)
        
        canny_image= cv2.Canny(image, int(max(0, (1.0 - 0.33) * medianImg)), int(min(255, (1.0 + 0.33) * medianImg)))
        comps,labels= cv2.connectedComponents(canny_image, connectivity=8)
        double_image=np.array(image,dtype=np.float64)
        # Gradient is defined as (change in y)/(change in x)
        [gx, gy] = np.gradient(double_image)
        #print(gy)
        #print(gx)
        values= np.sqrt(np.square(gy)+np.square(gx))#
        #print(value)
        
        #i=np.array(gx == 0,dtype=np.int32)   
        gx[gx == 0]=dummy_value
        #gy2 = np.gradient(gy)[1]  
        
        #consider angle range is always 360 degrees
        aarray=np.divide((np.arctan2(gy, gx) + np.pi) * 180., np.pi)
        #print(labels)
        for k in range(comps):
            xcoordinate, ycoordinate = np.where(labels==k)
            #print(ycoordinate)
            #print(xcoordinate)
            for j in range(xcoordinate.shape[0]):
                ypoint = ycoordinate[j]
                xpoint = xcoordinate[j]

                z = np.ceil(aarray[xpoint,ypoint]/degree)
                if z==0: 
                    numberOfBins=1
                if values[xpoint,ypoint]>0:
                    binaryHist[xpoint,ypoint] = z
                    binaryGrad[xpoint,ypoint] = values[xpoint,ypoint]
        #Looping on each level in pyramid
    #histb=binaryHist[0:490,0:490]
    #gradb=binaryGrad[0:490,0:490]
    #print(len(binaryHist))
    #print(len(binaryGrad))
    histb=binaryHist
    gradb=binaryGrad
    for k in range(numberOfBins):
        ind = histb==k
        pyramidarr = np.append(pyramidarr, np.sum(gradb[ind]))

    #higher levels
    for level in range(1, numberOfLevels+1):
        y = int(np.trunc(histb.shape[0]/(2**level)))
        x = int(np.trunc(histb.shape[1]/(2**level)))
        for yy in range(0, histb.shape[0]-y+1, y):
            for xx in range(0, histb.shape[1]-x+1, x):
                    #print(pyramidarr)
                binaryHist2 = histb[yy:yy+y,xx:xx+x]
                binaryGrad2 = gradb[yy:yy+y,xx:xx+x]

                for binofhist in range(numberOfBins):
                    ind = binaryHist2==binofhist
                    pyramidarr = np.append(pyramidarr, np.sum(binaryGrad2[ind], axis=0))
        
    if np.sum(pyramidarr)==0:
        return pyramidarr
    else:
        return pyramidarr/np.sum(pyramidarr)
 #############################################################################################################################       
def lpq(image, win_size=11):
    WIN_SIZE: int = win_size
    ALPHA: float = 1 / WIN_SIZE
    WIN_RADIUS: int = (WIN_SIZE - 1) // 2 

    x = np.arange(-WIN_RADIUS,WIN_RADIUS+1)[np.newaxis]
    w0 = np.ones_like(x)
    w1 = np.exp(-2*np.pi*x*ALPHA*1j)
    w2 = np.conj(w1)
    filterResp1 = convolve2d(convolve2d(image, w0.T, 'valid'),w1,'valid')
    filterResp2 = convolve2d(convolve2d(image, w1.T, 'valid'),w0,'valid')
    filterResp3 = convolve2d(convolve2d(image, w1.T, 'valid'),w1,'valid')
    filterResp4 = convolve2d(convolve2d(image, w1.T, 'valid'),w2,'valid')

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                    filterResp2.real, filterResp2.imag,
                    filterResp3.real, filterResp3.imag,
                    filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)
    LPQdesc = np.histogram(LPQdesc.flatten(),range(256))[0]

    return LPQdesc / LPQdesc.max()
 #############################################################################################################################   
def lpq_plus(image, win_size = 11, q_levels=8):
    WIN_SIZE: int = win_size
    ALPHA: float = 1 / WIN_SIZE
    CONV_MODE:str = 'valid'
    WIN_RADIUS: int = (WIN_SIZE - 1) // 2 
    Q_LEVELS: int = q_levels
    FUND_ANGLE: float = (2 * np.pi) / Q_LEVELS
    VOTE_ALPHA: int = 2

    x = np.arange(-WIN_RADIUS,WIN_RADIUS+1)[np.newaxis]
    w0 = np.ones_like(x)
    w1 = np.exp(-2*np.pi*x*ALPHA*1j)
    w2 = np.conj(w1)

    ANGLES = [FUND_ANGLE * i for i in range(Q_LEVELS)]

    # Applying filters and getting response
    filterResp1: np.ndarray = convolve2d(convolve2d(image, w0.T, 'valid'),w1,'valid') # (0, a)
    filterResp2: np.ndarray = convolve2d(convolve2d(image, w1.T, 'valid'),w0,'valid') # (a, 0)
    filterResp3: np.ndarray = convolve2d(convolve2d(image, w1.T, 'valid'),w1,'valid') # (a, a)
    filterResp4: np.ndarray = convolve2d(convolve2d(image, w1.T, 'valid'),w2,'valid') # (a, -a)

    # Local Phases of the Responses
    theta1 = np.arctan2(filterResp1.imag, filterResp1.real)
    theta2 = np.arctan2(filterResp2.imag, filterResp2.real)
    theta3 = np.arctan2(filterResp3.imag, filterResp3.real)
    theta4 = np.arctan2(filterResp4.imag, filterResp4.real)

    vote1 = np.array(list(map(lambda theta: (np.cos(theta1 - theta)**VOTE_ALPHA).sum(), ANGLES)))
    vote2 = np.array(list(map(lambda theta: (np.cos(theta2 - theta)**VOTE_ALPHA).sum(), ANGLES)))
    vote3 = np.array(list(map(lambda theta: (np.cos(theta3 - theta)**VOTE_ALPHA).sum(), ANGLES)))
    vote4 = np.array(list(map(lambda theta: (np.cos(theta4 - theta)**VOTE_ALPHA).sum(), ANGLES)))
    lpq_desc = np.concatenate((vote1, vote2, vote3, vote4))

    # Normailze the data
    return  lpq_desc / lpq_desc.max()

    import cv2
import numpy as np

##############################################################################################################################

# how to use gabour filters 
##### the image needs to be 1- gray image 2- cropped (face only)  3- reszie the croped image to (128*96)
##### filters to be bulit the theta needs to be with value 8  and scales  = [3,5,7,9,11]

class GabourFeatures:

    def __init__(self, theta=8, scales= [3,5,7,9,11]):
        filters = []
        psi = np.pi/2.0
        gamma = 0.3
        lamda = 5
        sigma = 3
        "preparing filters with the aboved parameters "
        for i in range(theta):
            theta = ((i+1)*1.0 / theta) * np.pi
            for scale in scales:
                kernel = cv2.getGaborKernel((scale, scale), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
                filters.append(kernel)
        self.filters = filters

    def compute(self, image):
        "apply the filters to the image to get the features"
        features = []
        for filter in self.filters:
            filtered= cv2.filter2D(image, cv2.CV_8UC3, filter)
            feature = filtered.reshape(-1)
            features.extend(feature)
        return features

#############################################################################################################################

