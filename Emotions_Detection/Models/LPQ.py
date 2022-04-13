import numpy as np
from scipy.signal import convolve2d

def lpq(img,winSize=3,mode='nh'):
    STFTalpha = 1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS = (winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)           # Convert np.image to double
    r = (winSize-1)/2               # Get radius from window size
    x = np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    
    #  Basic STFT filters
    w0 = np.ones_like(x)
    print(w0.shape)
    w1 = np.exp(-2*np.pi*x*STFTalpha*1j)
    w2 = np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode),w1,convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode),w0,convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode),w1,convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])
    print(freqResp.shape)
    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc/LPQdesc.sum()

    #print(LPQdesc)
    return LPQdesc


def lpq_plus(image, win_size=11, q_levels=8):
    '''Takes an image and applys LPQ_plus filters and return the resulting histogram\n
        '''
    # Inital Variables(Constants)
    WIN_SIZE: int = win_size
    ALPHA: float = 1 / WIN_SIZE
    CONV_MODE:str = 'valid'
    WIN_RADIUS: int = (WIN_SIZE - 1) // 2 
    Q_LEVELS: int = q_levels
    FUND_ANGLE: float = (2 * np.pi) / Q_LEVELS
    VOTE_ALPHA: int = 2

    # Creating Filters
    x = np.arange(-WIN_RADIUS,WIN_RADIUS+1)[np.newaxis]
    w0 = np.ones_like(x)
    w1 = np.exp(-2*np.pi*x*ALPHA*1j)
    w2 = np.conj(w1)
    
    # Applying filters and getting response
    filterResp1: np.ndarray = convolve2d(convolve2d(image, w0.T, CONV_MODE),w1,CONV_MODE) # (0, a)
    filterResp2: np.ndarray = convolve2d(convolve2d(image, w1.T, CONV_MODE),w0,CONV_MODE) # (a, 0)
    filterResp3: np.ndarray = convolve2d(convolve2d(image, w1.T, CONV_MODE),w1,CONV_MODE) # (a, a)
    filterResp4: np.ndarray = convolve2d(convolve2d(image, w1.T, CONV_MODE),w2,CONV_MODE) # (a, -a)

    # Local Phases of the Responses
    theta1 = np.arctan2(filterResp1.imag, filterResp1.real)
    theta2 = np.arctan2(filterResp2.imag, filterResp2.real)
    theta3 = np.arctan2(filterResp3.imag, filterResp3.real)
    theta4 = np.arctan2(filterResp4.imag, filterResp4.real)

    # Phases Quantization by vote
    angles = [FUND_ANGLE * i for i in range(Q_LEVELS)]

    vote1 = np.array(list(map(lambda theta: (np.cos(theta1 - theta)**VOTE_ALPHA).sum(), angles)))
    vote2 = np.array(list(map(lambda theta: (np.cos(theta2 - theta)**VOTE_ALPHA).sum(), angles)))
    vote3 = np.array(list(map(lambda theta: (np.cos(theta3 - theta)**VOTE_ALPHA).sum(), angles)))
    vote4 = np.array(list(map(lambda theta: (np.cos(theta4 - theta)**VOTE_ALPHA).sum(), angles)))

    # Normailze the data
    vote1 = vote1 / vote1.max()
    vote2 = vote2 / vote2.max()
    vote3 = vote3 / vote3.max()
    vote4 = vote4 / vote4.max()
    lpq_desq = np.concatenate((vote1, vote2, vote3, vote4))

    return lpq_desq