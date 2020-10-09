import os
import sys
import glob
import time
import multiprocessing
from optparse import OptionParser, SUPPRESS_HELP

# Loading third party packages
import numpy as np


def recordingSize(orig_ny, orig_nx , inner_ny ,inner_nx , name , mname ):
    # Recording originSize and downscaleSize
    with h5py.File( mname , 'w') as h5F:
        dset = h5F.create_dataset('meta' , (4,) , dtype = np.float ,compression = 'gzip')
        dset[:] = np.asarray([ orig_ny, orig_nx , inner_ny ,inner_nx ])
    return 0

def recordingMaskSize(orig_ny, orig_nx , inner_ny ,inner_nx , name , mname , promap , maskmap):
    # Recording originSize and downscaleSize
    with h5py.File( mname , 'w') as h5F:
        #print(mname)
        dset = h5F.create_dataset('meta' , (4,) , dtype = np.float ,compression = 'gzip')
        dset[:] = np.asarray([ orig_ny, orig_nx , inner_ny ,inner_nx ])
        dset2 = h5F.create_dataset('promap' , promap.shape , dtype = np.float ,compression = 'gzip')
        dset2[:]= promap
        dset3 = h5F.create_dataset('maskmap' , maskmap.shape , dtype = np.int ,compression = 'gzip')
        dset3[:]= maskmap        
    return 0

def segFCNt(_innerAngPixel = 15 , _angPixel = 1.3 , _nSize = 256 , model_adopt = 'pre_train_model.h5' , totalStage = 16):
    def scale(x):
        y = K.max(x) ** -1
        x = x * y
        return x
    def scale_output_shape(input_shape):
        shape = list(input_shape)
        return tuple(shape)
    
    _innerNSize = _nSize
    input_ap = Input(shape=(_innerNSize,_innerNSize,1),name = 'input_ap')
    input_mk = Input(shape=(_innerNSize,_innerNSize,1),name = 'input_mk')
    conv1 = Conv2D(32, (9,9),padding='same',activation='relu',name='conv1')(input_ap)
    conv2_1 = Conv2D(32, (9,9),padding='same',activation='relu',name='conv2_1')(conv1)
    conv2_2 = Conv2D(32, (9,9),padding='same',activation='linear',name='conv2_2')(conv2_1)
    conv2 = LeakyReLU(alpha=0.0)(add([conv1,conv2_2],name='conv2'))
    conv3_1 = Conv2D(32, (9,9),padding='same',activation='relu',name='conv3_1')(conv2)
    conv3_2 = Conv2D(32, (9,9),padding='same',activation='linear',name='conv3_2')(conv3_1)
    conv3 = LeakyReLU(alpha=0.0)(add([conv2,conv3_2],name='conv3'))
    deconv1_1 = Conv2DTranspose(32, kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu',name='deconv1_1')(conv3)
    deconv1 = LeakyReLU(alpha=0.0)(add([conv2 , deconv1_1],name='deconv1'))
    deconv2 = Conv2DTranspose(32, kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu',name='deconv2')(deconv1)
    deconv3 = Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu',name='deconv3')(deconv2)
    masked = Multiply(name='masked')([deconv3 , input_mk])
    scaled = masked
    scaledf= Reshape((_innerNSize * _innerNSize , 1),name='reshape')(scaled)
    model_ap = Model(inputs = [input_ap, input_mk], outputs = scaledf)
    model_ap.load_weights(model_adopt ,  by_name=True)
    return model_ap

def extract_format(extformat):
    switcher = {
        'star' : ['\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2','star'],
        'csv' : [None,None],
    }
    return switcher.get(extformat, None)
   

def cvopen(img , cvkSize):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,( cvkSize , cvkSize )),iterations=1)

def cvclose(img , cvkSize):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,( cvkSize , cvkSize )),iterations=1)
    
def npcircle(radius):
    xx , yy = np.meshgrid(np.linspace(-radius , radius , radius * 2 + 1 ) , np.linspace(-radius , radius , radius * 2 + 1 ))
    zz = xx ** 2 + yy ** 2 <= radius ** 2
    return zz.astype('uint8')

# 1-D k-means
def k2means(inputSerial, k = 2, iters = 3):
    lenSer = len(inputSerial)
    sites = np.zeros(k,dtype=float)
    for i in range(k):
        sites[i] = inputSerial[np.random.randint(lenSer)]
    iterss = 0
    sites_new = np.zeros_like(sites,dtype=np.float)
    sites_new_count = np.zeros_like(sites,dtype=np.float)
    while iterss < iters:
        iterss += 1
        for i in range(lenSer):
            label = np.argmin(abs(inputSerial[i] - sites))
            sites_new[label] += inputSerial[i]
            sites_new_count[label] += 1
        for i in range(k):
            sites[i] = sites_new[i]/sites_new_count[i]
        sites_new *= 0
        sites_new_count *= 0
    return sites

# rewindPM
def lo_thres_calc(mask, signi = 0.01):
    ii = 0
    maskSer = np.sort(mask.flatten())[::-1]
    lenSer = len(maskSer)
    # significance
    sigValue = signi * len(maskSer) / 100
    thresLst= []
    for i in np.arange(0,1,0.01)[::-1]:
        while (ii<len(maskSer)) and (maskSer[ii]>i):
            ii += 1
        thresLst.append(ii)
    
    serLst2 = np.diff(np.diff(thresLst))
    # Oth find
    currentSgn = [1]
    ii = 0
    lo_thresholdP_o = -1
    for i in serLst2:
        if i > sigValue:
            if currentSgn[-1] != 1:
                currentSgn.append(1)
        if i <-sigValue:
            if currentSgn[-1] != -1:
                currentSgn.append(-1)
        if currentSgn == [1,-1,1]:
            lo_thresholdP_o = 1 - ii * 1e-2
            break;
        ii += 1
    
    # Reverse find
    currentSgn = [1]
    ii = 0
    lo_thresholdP_r = -1
    for i in serLst2[::-1]:
        if i <-sigValue:
            if currentSgn[-1] != 1:
                currentSgn.append(1)
        if i > sigValue:
            if currentSgn[-1] != -1:
                currentSgn.append(-1)
        if currentSgn == [1,-1,1]:
            lo_thresholdP_r = ii * 1e-2 + 2e-2
            break;
        ii += 1
    lo_threshold_R = np.asarray([lo_thresholdP_o, lo_thresholdP_r])
    return lo_threshold_R, lo_threshold_R < 0.1

def adpmassO(inMap, inSize=9 , binSize=30 , conv = 5):
    ctp = tp.locate( inMap , inSize ,  minmass = 0)
    hist = np.histogram(ctp['mass'],bins= binSize)
    summ = np.convolve([1,] * conv , hist[0] , mode='same')
    velo = np.diff(summ)
    acce = np.diff(velo)
    ii = 0
    iiT= len(velo)
    while (ii < iiT) and (velo[ii] > 0):
        ii += 1
    while (ii < iiT) and (velo[ii] <= 0):
        ii += 1
    if (ii == iiT):
        return 0
    else:
        print(hist[1][ii + 1])
        return hist[1][ii + 1]
    
def adpmass(inMap, inSize=9 , binSize=30 , conv = 5 , thres = 5 ):
    ctp = tp.locate( inMap , inSize ,  minmass = 0)
    hist = np.histogram(ctp['mass'],bins= binSize)
    summ = np.convolve([1,] * conv , hist[0] , mode='same')
    velo = np.diff(summ)
    acce = np.diff(velo)
    ii = 0
    iiT= len(velo)
    while (acce[ii]// thres > 0) and (ii < iiT):
        ii += 1
    while (acce[ii]// thres < 0) and (ii < iiT):
        ii += 1
    while (acce[ii]// thres > 0) and (ii < iiT):
        ii += 1
    if (ii == iiT):
        return 0
    else:
        print(hist[1][ii])
        return hist[1][ii]

def micGen(name , gauFactor , mname , dataset_angPixel , inputSize = 1024 , reverseI = True , highPass = 500 , hard = True , lowerBound = 0 , upperBound = 3, DEBUG=False, global_TrackParticleSize_actual = 9, hard_dust_removal=True , cuttingedge_lowD=0 , cuttingedge_highD=-1 , cscale = 4):
    #Input Map
    inputMap = np.zeros( (inputSize , inputSize , 1) , dtype = np.float32)

    with mrcfile.open(name , 'r' , permissive=True) as mrc:
        origMap  = mrc.data
        origMapNC= mic_preprocess.nmlizeC(iput = origMap , cscale = 4)
    downScale , GauSigma , kSize = gauFactor

    # checkpoint mrcread
    if global_timer:                        
        timeseries_end1.append(time.time())

    filterMap = cv2.GaussianBlur( origMapNC , (kSize , kSize) , GauSigma) #/ 0.1 #ampCont

    if reverseI == True:
        filterMap = - filterMap

    inner_ny , inner_nx , actualAngPixel_yx = mic_preprocess.downScaling(input_shape = origMapNC.shape, 
                                                                         input_innerAngPixel = dataset_angPixel,
                                                                         target_innerAngPixel = global_innerAngPixel)
    downMap   = cv2.resize( filterMap , (inner_nx , inner_ny))
    
    BPMaskS = mic_preprocess.genLPMaskS( inner_ny , inner_nx , np.mean(actualAngPixel_yx) * 2 , highPass , np.mean(actualAngPixel_yx) , 0 , 1)
    BPMap   = mic_preprocess.LPFilterS( downMap , BPMaskS , 0 , 1)#Scale) #/ ny / nx * 4
    #plt.figure();plt.imshow(BPMaskS[:,:,0])

    dataFeeder_relu = mic_preprocess.nmlizeC(iput = BPMap , cscale = cscale)
    
    # NIPS 2016 red
    BPMap_orig = dataFeeder_relu.copy()    

    if DEBUG:
        #plt.figure();plt.imshow(BPMap,'gray',vmin=0.)
        print(BPMap.shape)
        print(np.mean(BPMap_orig),np.std(BPMap_orig),np.mean(BPMap),np.std(BPMap))
        
    dataFeeder_relu_std = 1
    dataFeeder_relu[dataFeeder_relu < (lowerBound * dataFeeder_relu_std)] = lowerBound * dataFeeder_relu_std
    dataFeeder_relu[dataFeeder_relu > (upperBound * dataFeeder_relu_std)] = upperBound * dataFeeder_relu_std
    dataFeeder_relu -= lowerBound * dataFeeder_relu_std
    dataFeeder_relu /= (upperBound - lowerBound) * dataFeeder_relu_std
    BPMap = dataFeeder_relu

    inputMap[ :inner_ny , :inner_nx , 0] += BPMap#downMap

    if hard_dust_removal==False:
        mask_final = np.ones_like(downMap)
        mask_final[cuttingedge_lowD:cuttingedge_highD , cuttingedge_lowD:cuttingedge_highD] = 0

        mask_finalt= np.zeros((cuttingedge_highD-cuttingedge_lowD , cuttingedge_highD-cuttingedge_lowD),dtype=bool)

        masky, maskx = mask_finalt.shape
        mask_final2= np.ones((inputSize,inputSize),dtype= bool)
        mask_final2[cuttingedge_lowD:cuttingedge_lowD+masky , cuttingedge_lowD:cuttingedge_lowD+maskx] *= mask_finalt

        # checkpoint preprocess
        if global_timer:       
            timeseries_end2.append(time.time())
            
        inputMap[ :inner_ny , :inner_nx , 0] -= BPMap
        inputMap[ :inner_ny , :inner_nx , 0] += BPMap_orig
        
        recordingMaskSize(origMapNC.shape[0], origMapNC.shape[1] , inner_ny , inner_nx , name , mname = mname , promap = inputMap , maskmap = mask_final)

        return [inputMap,downMap,name.split('/')[-1].split('.')[0],mask_final,mask_final2]
    
    # Mask calculation
    # loading half-preprocessed map    
    tp_imgdo = downMap[:inner_ny , :inner_nx][cuttingedge_lowD:cuttingedge_highD , cuttingedge_lowD:cuttingedge_highD]
    tp_imgt = BPMap[:inner_ny , :inner_nx][cuttingedge_lowD:cuttingedge_highD , cuttingedge_lowD:cuttingedge_highD]

    # init hi/low-pass filter
    downScaleL, GauSigmaL , kSizeL = mic_preprocess.gauScaling(input_innerAngPixel = global_innerAngPixel , lowThres = highPass) #(cas9=50)
    gauFactorL = [downScaleL , GauSigmaL , kSizeL]

    mask   = mic_preprocess.gauLPFilter( tp_imgdo , gauFactorL )
    mask2  = mic_preprocess.gauLPFilter( tp_imgt  , gauFactorL )
    
    # calculation mask
    mask_ref = mask - mask2
    lo_thresholdP,thresHard = lo_thres_calc(mask_ref)
    
    # recalulation on BPMap if downMap failed
    if np.all(thresHard):#lo_thresholdP == -1:
        mask_ref = mask2
        lo_thresholdP,thresHard = lo_thres_calc(mask_ref)
    
    if np.all(thresHard):#lo_thresholdP < 0.1:
        lo_thresholdP = 1
        mask_finalt = (mask_ref > lo_thresholdP)
    else:
        # inconsistency-check for upside-down/downside-up peak finding
        lo_thresholdPmin = np.min(lo_thresholdP[~thresHard]) 
        lo_thresholdPmax = np.max(lo_thresholdP[~thresHard]) 
        if np.all(lo_thresholdP > 0.35):
            lo_thresholdP = lo_thresholdPmax
        elif (lo_thresholdPmax / lo_thresholdPmin > 2.5):# and (lo_thresholdPmax >= 0.5):
            lo_thresholdP = lo_thresholdPmax
        else:
            lo_thresholdP = lo_thresholdPmin
        mask_fini = (mask_ref > lo_thresholdP).astype('uint8')
        mask_fini = cv2.dilate(mask_fini , npcircle( global_TrackParticleSize_actual ) , iterations = 1)

        im2, contours, hierarchy = cv2.findContours(mask_fini, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
        im3 = np.zeros_like(im2)
        # maximum_size
        thresArea = 225
        contoursFinal = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > thresArea:
                contoursFinal.append(cnt)
        cvdraw = cv2.drawContours(im3, contoursFinal, -1 , 1 , cv2.FILLED)

        mask_finalt= im3 > 0

    mask_final = np.ones((inner_ny, inner_nx),dtype = bool)
    mask_final[cuttingedge_lowD:cuttingedge_highD , cuttingedge_lowD:cuttingedge_highD] *= mask_finalt
    
    
    masky, maskx = mask_finalt.shape
    mask_final2= np.ones((inputSize,inputSize),dtype= bool)
    mask_final2[cuttingedge_lowD:cuttingedge_lowD+masky , cuttingedge_lowD:cuttingedge_lowD+maskx] *= mask_finalt
    
    # redo previous normalization based on new global masking
    origMapNC= mic_preprocess.nmlizeM(iput = origMap , mask = mask_final, cscale = 4)
    filterMap = cv2.GaussianBlur( origMapNC , (kSize , kSize) , GauSigma) #/ 0.1 #ampCont

    if reverseI == True:
        filterMap = - filterMap

    inner_ny , inner_nx , actualAngPixel_yx = mic_preprocess.downScaling(input_shape = origMapNC.shape, 
                                                                         input_innerAngPixel = dataset_angPixel,
                                                                         target_innerAngPixel = global_innerAngPixel)
    downMap   = cv2.resize( filterMap , (inner_nx , inner_ny))
    
    BPMaskS = mic_preprocess.genLPMaskS( inner_ny , inner_nx , np.mean(actualAngPixel_yx) * 2 , highPass , np.mean(actualAngPixel_yx) , 0 , 1)
    BPMap   = mic_preprocess.LPFilterS( downMap , BPMaskS , 0 , 1)#Scale) #/ ny / nx * 4

    dataFeeder_relu = mic_preprocess.nmlizeM(iput = BPMap , mask = mask_final, cscale = 4)

    if DEBUG:
        #plt.figure();plt.imshow(BPMap,'gray',vmin=0.)
        print(np.mean(dataFeeder_relu),np.median(dataFeeder_relu),np.std(dataFeeder_relu))

    BPMap = dataFeeder_relu

    # overwrite BPMap
    inputMap[ :inner_ny , :inner_nx , 0] = BPMap#downMap
    
    # Recording originSize and downscaleSize and preprocessed/masked map
    recordingMaskSize(origMapNC.shape[0], origMapNC.shape[1] , inner_ny , inner_nx , name , mname = mname , promap = inputMap , maskmap = mask_final)
    
    # checkpoint preprocess
    if global_timer:       
        timeseries_end2.append(time.time())
    return [inputMap,downMap,name.split('/')[-1].split('.')[0],mask_final,mask_final2]

def track_part(pred_test, downMap, mediumName, \
               fcn_mapSize, \
               numClass, \
               cuttingedge_lowD , cuttingedge_highD, \
               hard_dust_removal, \
               gauFactorET, \
               global_lo_thresmin,\
               global_TrackParticleSize_actual,\
               TrackParticleSize,\
               TrackMinMass,\
               pickScale,\
               output_PATH,\
               job_suffix,\
               header,\
               nameMIC,\
               nameC,\
               currentImage,\
               TotalImage,\
               thresProb,\
               mask_prep,\
               global_adpmass,\
               global_enhance):
    # append segmap

    with h5py.File( mediumName , 'a') as h5F:
        dset = h5F.create_dataset('fcn' , (fcn_mapSize,fcn_mapSize , numClass) , dtype = np.float ,compression = 'gzip')
        dset[:] = pred_test[0,:].reshape(fcn_mapSize,fcn_mapSize , numClass)

    # load origal size
    with h5py.File( mediumName , 'r') as h5F:
        orig_ny , orig_nx , ds_ny ,ds_nx = h5F['meta'][:]
        tp_map = h5F['fcn'][:np.int(ds_ny) , :np.int(ds_nx) , numClass - 1][:]
        scale_ny = orig_ny / ds_ny; scale_nx = orig_nx / ds_nx

    tpy, tpx = np.asarray([ds_ny ,ds_nx]).astype(np.int)


    if hard_dust_removal:
        mask_final = mask_prep.astype('uint8')
    else:
        mask_final= np.ones_like(tp_map[:,:],dtype='uint8')

    # checkpoint dust hide
    if global_timer:   
        timeseries_end4.append(time.time())        

    if global_enhance:
        amplitute_avg = np.mean(tp_map[tp_map>0]) + 3 * np.std(tp_map[tp_map>0])
    else:
        amplitute_avg = 1
    
    tp_mapRS = cv2.resize(tp_map / amplitute_avg , (tpx * pickScale, tpy * pickScale) )

    if global_adpmass:
        curr_adpmass = adpmass( tp_mapRS , inSize = TrackParticleSize)
        if curr_adpmass < TrackMinMass:
            print('curr_adpmass smaller than TrackMinMass, use bigger one')
            curr_adpmass = TrackMinMass
    else:
        curr_adpmass = TrackMinMass
    
    coorTrackPYnpfit = tp.locate( tp_mapRS , TrackParticleSize , minmass = curr_adpmass )#, maxsize = TrackParticleSize )

    massPick = coorTrackPYnpfit['mass']
    sizePick = coorTrackPYnpfit['size']

    coorTrackPY = coorTrackPYnpfit.copy()
    coorTrackPY[['x','y']] /= pickScale
    coorTrackPY.loc[:,'prob']=[tp_mapRS[getattr(coor,'y'),getattr(coor,'x')] for coor in np.floor(coorTrackPY[['x','y']]).astype('int').itertuples()]
    coorTrackPY['y'] *= scale_ny
    coorTrackPY['x'] *= scale_nx

    coorTrackPYtango = coorTrackPY
    coorTrackPYdown = coorTrackPYtango.copy()#[['x','y']]
    coorTrackPYdown['x'] /= scale_nx
    coorTrackPYdown['y'] /= scale_ny

    coorTrackPYnpfit.loc[:,'prob']=[tp_map[getattr(coor,'y'),getattr(coor,'x')] for coor in np.floor(coorTrackPYdown[['x','y']]).astype('int').itertuples()]

    coorTrackPY.loc[:,'prob'] = coorTrackPYnpfit.loc[:,'prob']

    SkipPickNum = 0


    with open('./%s/%s_%s.star' %(output_PATH , nameC[:-4] , job_suffix), 'w') as coorF:
        coorFa = open('./%s/%s_%s_param.star' %(output_PATH , nameC[:-4] , job_suffix), 'w')
        coorTrack = coorTrackPY.copy()#[['x','y']]
        # STAR HEADING
        coorF.write( header )
        coorFa.write( header )
        SkipPickNum = 0
        for coor in coorTrack.itertuples():
            coor_x = getattr(coor,'x')
            coor_y = getattr(coor,'y')
            if (getattr(coor,'prob') < thresProb):
                SkipPickNum += 1
                continue
            coorF.write( '\n' + str( np.floor(coor_x).astype(np.int) ).rjust( 6 ) + '.000000' + str( np.floor(coor_y).astype(np.int) ).rjust( 6 ) + '.000000 ' )
            coorFa.write( '\n' + str( np.floor(coor_x).astype(np.int) ).rjust( 6 ) + '.000000' + str( np.floor(coor_y).astype(np.int) ).rjust( 6 ) + '.000000 ' )
            for item in list(coorTrackPY.columns.values):
                coorFa.write('%5.6f ' %getattr(coor, item))
        coorFa.close()

    # checkpoint location
    if global_timer:
        timeseries_end.append(time.time())
            
    print('Coordinates extraction on %s finished\t%8d found\t\tTimespan%8.3fs\t[%5d / %5d ]' %(nameMIC, len(coorTrackPY) - SkipPickNum , timeseries_end[-1] - timeseries_begin[-1], currentImage, TotalImage))    
    tTotalPickNum = len(coorTrackPY) - SkipPickNum    
    return tTotalPickNum


def void_picking(optionsk,argsk):
   
    model_adopt = optionsk.model
    job_suffix = optionsk.job_suffix
    ext_format = optionsk.ext_format
    fcn_mapSize = (np.ceil(optionsk.img_size * optionsk.angpixel / global_innerAngPixel / global_downScale) * global_downScale).astype('int')#np.max(optionsk.fcn_mapsize,)
    dataset_angPixel = optionsk.angpixel
    dataPath = optionsk.data_path
    output_PATH = optionsk.output_path
    hard_dust_removal = optionsk.dust_removal
    global_TrackMinMass = optionsk.mass_min
    global_adpmass = optionsk.adpmass
    global_enhance = optionsk.enhance
    
    global_cvkSize = optionsk.cvk_size
    pickScale = optionsk.resample_rate
  
    global_TrackParticleSize_actual = np.floor(optionsk.aperture / global_innerAngPixel).astype('int')
    global_TrackParticleSize = global_TrackParticleSize_actual * pickScale + 1

    saturate_rate = optionsk.saturate_rate
    saturate_half = saturate_rate / 2

    cuttingedge_low   =  optionsk.edge_cut
    cuttingedge_high  =  optionsk.img_size - cuttingedge_low
    cuttingedge_lowD  =  np.ceil(cuttingedge_low / (optionsk.img_size / fcn_mapSize )).astype(np.int)
    cuttingedge_highD =  fcn_mapSize - cuttingedge_lowD

    thresProb = optionsk.prob_thres
    cscale = optionsk.cscale
    
    file_pattern = optionsk.file_pattern
    
    nameList = glob.glob('%s/%s' %(dataPath,file_pattern))
    
    TrackParticleSize = global_TrackParticleSize#5
    TrackMinMass = global_TrackMinMass#1
    
    if not os.path.exists(output_PATH):
        os.mkdir(output_PATH)

    # coor info
    #ext_format = 'star'
    header, prefix = extract_format(ext_format)

    # preprocess
    downScale , GauSigma , kSize = mic_preprocess.gauScaling(input_innerAngPixel = dataset_angPixel, lowThres = 30)
    gauFactor = [downScale , GauSigma , kSize]
    
    # lo-freq depression
    lodep_thres = optionsk.lo_dep
    downScaleET , GauSigmaET , kSizeET = mic_preprocess.gauScaling(input_innerAngPixel = global_innerAngPixel, lowThres = lodep_thres) #(cas9=50)
    gauFactorET = [downScaleET , GauSigmaET , kSizeET]
    

    # model load
    model_micSEG = segFCNt(_innerAngPixel = global_innerAngPixel , _nSize = fcn_mapSize , model_adopt = model_adopt)

    # do filtering/segmentation work
    TotalPickNum = 0

    # timer init
    import time
    global timeseries_begin,timeseries_end1,timeseries_end2,timeseries_end3,timeseries_end4,timeseries_end
    timeseries_begin = []
    timeseries_end1 = []
    timeseries_end2 = []
    timeseries_end3 = []
    timeseries_end4 = []
    timeseries_end = []

    TotalImage = len(nameList)
    currentImage = 0
    coreNum = optionsk.core_num
    
    # file iteration
    for name_ii in range(0, TotalImage, coreNum):#range(TotalImage//coreNum + np.int(np.ceil(TotalImage / coreNum))):
        nameList_t = nameList[name_ii: name_ii + coreNum]
        currentImage += len(nameList_t)
        timeseries_begin.append(time.time())
        
        # batch inputMap/downMap
        batchPool = multiprocessing.Pool()
        batchRes = []
        
        for name in nameList_t:
            dataPath = name[:-len(name.split('/')[-1])-1]
            nameC = name[len(dataPath) + 1:]
            nameMIC = nameC[:-4]
            nameFile = dataPath + '/' + nameC
            mediumName = '%s/%s_%s.h5' %(output_PATH , nameC[:-4] , job_suffix)
            batchResS = batchPool.apply_async(micGen, args=(nameFile,\
                                                            gauFactor,\
                                                            mediumName ,\
                                                            dataset_angPixel,\
                                                            fcn_mapSize,\
                                                            True, 500, True, 0 , 3 ,False, \
                                                            global_TrackParticleSize_actual,\
                                                            hard_dust_removal , cuttingedge_lowD , cuttingedge_highD,\
                                                            cscale ))
            batchRes.append(batchResS)
            
        batchPool.close();batchPool.join()
        
        batch_inputMap = []
        batch_inputFCN = []
        batch_downMap  = []
        batch_nameLst  = []
        batch_mask     = []
        batch_mask2    = []
        for batchRess in batchRes:
            batchResss = batchRess.get()
            batch_inputMap.append(batchResss[0])
            batch_downMap.append(batchResss[1])
            batch_nameLst.append(batchResss[2])
            batch_mask.append(batchResss[3])
            batch_mask2.append(batchResss[4])
                  
        batch_inputMap = np.asarray(batch_inputMap)
        batch_mask_in  = 1 - np.asarray(batch_mask2)[:,:,:,np.newaxis]
       
        # start actual prediction
        pred_test = model_micSEG.predict([batch_inputMap, batch_mask_in] , batch_size=1, verbose=0, steps=None)
        
        # checkpoint segement
        if global_timer:   
            timeseries_end3.append(time.time())
        
        batchPool = multiprocessing.Pool()
        
        batchRes = []
        ii = 0
        for ii in range(len(nameList_t)):
            name = nameList_t[ii]
            dataPath = name[:-len(name.split('/')[-1])-1]
            nameC = name[len(dataPath) + 1:]
            nameMIC = nameC[:-4]
            nameFile = dataPath + '/' + nameC
            mediumName = '%s/%s_%s.h5' %(output_PATH , nameC[:-4] , job_suffix)
            batchResS = batchPool.apply_async(track_part, args=(pred_test[ii][np.newaxis,:,:],\
                                                                batch_downMap[ii],\
                                                                mediumName, \
                                                                fcn_mapSize, \
                                                                numClass, \
                                                                cuttingedge_lowD ,cuttingedge_highD, \
                                                                hard_dust_removal, \
                                                                gauFactorET, \
                                                                global_lo_thresmin,\
                                                                global_TrackParticleSize_actual,\
                                                                TrackParticleSize,\
                                                                TrackMinMass,\
                                                                pickScale,\
                                                                output_PATH,\
                                                                job_suffix,\
                                                                header,\
                                                                nameMIC,\
                                                                nameC,\
                                                                name_ii + ii,\
                                                                TotalImage,\
                                                                thresProb,\
                                                                batch_mask[ii],\
                                                                global_adpmass,\
                                                                global_enhance))
            batchRes.append(batchResS)
            
        batchPool.close();batchPool.join()
                
        for batchRess in batchRes:
            batchResss = batchRess.get()
            TotalPickNum += batchResss
       
    print('Coordinates extraction finished\t\t%8d TotalPickNum found' %TotalPickNum)
    
    # record picking parameters info
    with open('./%s/%s_columns.star' %(output_PATH , job_suffix), 'w') as coorFb:
        for item in list(['x' 'y' 'mass' 'size' 'ecc' 'signal' 'raw_mass' 'ep']+['prob' + 'mask']):#list(coorTrackPY.columns.values):
            coorFb.write('%s\t' %item)
            
    # record checkpoint time
    with h5py.File('./%s/%s_%s.h5' %(output_PATH , 'timeseries' , job_suffix), 'w') as h5t:
        h5t['begin'] = np.asarray(timeseries_begin)
        h5t['end1'] = np.asarray(timeseries_end1)
        h5t['end2'] = np.asarray(timeseries_end2)
        h5t['end3'] = np.asarray(timeseries_end3)
        h5t['end4'] = np.asarray(timeseries_end4)
        h5t['end'] = np.asarray(timeseries_end)
        
if __name__ == '__main__':
    progName = os.path.basename(sys.argv[0])
#    usage = progName + """ <train|valid|pick> [options]
    usage = progName + """ [options]
    
    PARSED (PARticle SEgmentation Detector)

    PARSED is a deep-learning model that reads a list of MRC files of cryo-EM micrographs, 
    and then automatically picks particles of biological macromolecules in these micrographs. 
    The picked particles could be directly imported into 3D reconstruction programs such as 
    cryoSPARC or RELION. 

    Reference:
    R. Yao, J. Qian, Q. Huang. Deep-learning with synthetic data enables automated picking 
    of cryo-EM particle images of biological macromolecules (To be published).
    """
    
    # pre-defined
    global_innerAngPixel = 15.0
    global_downScale = 4
    
    numClass = 1
    global_timer = True
    global_lo_thresmin = 0.10
    
    parser = OptionParser(usage , version='PARSED 0.0.1')
    
    #Job Parameters
    parser.add_option("--model", metavar='./pre_train_model.h5', default='./pre_train_model.h5', type='string', help = "file name of used model (default pre_train_model.h5)")
    parser.add_option("--data_path", metavar='PATH_OF_MICROGRAPHS', default=None, type='string', help= "path for target micrograph sets")
    parser.add_option("--output_path", metavar="PATH_OUTPUT", default=None, type='string', help="path for output location")
    parser.add_option("--file_pattern", metavar="stack_*_00??.mrc", default = None, type='string', help = "regular expression for matching micrograph files")
    parser.add_option("--job_suffix", metavar='demo0', default='demo0', type='string', help = "output coordinates files suffix (default demo0)")
    parser.add_option("--angpixel", metavar=1.0, default=None, type=np.float, help = "micrograph sample-rate (default 1.0 A/pixel)")
    parser.add_option("--img_size", metavar=4096, default=None, type=np.int, help = "long edge size of micrograph files (default 4096)")
    parser.add_option("--edge_cut", metavar=0, default=0, type=np.int, help = "edge crop size for micrograph files (default 0)")
    
    # Dust Removal
    parser.add_option("--lo_dep", metavar=200., default=200., type=np.float, help = "dust depression filter size (default 200 A)")

    # Picking Parameters
    parser.add_option("--core_num", metavar=1, default=1, type=np.int, help = "number of processes for picking (default 1)")
    parser.add_option("--cscale", metavar=4, default=4, type=np.int, help = "central range for localized normalization (default 4)")
    parser.add_option("--resample_rate", metavar=2, default=2, type=np.int, help = "rescaling rate for particle picking (default 2)")
    parser.add_option("--aperture", metavar=128, default=128, type=np.int, help = "detection aperture for particle (default 128 A)")
    parser.add_option("--mass_min", metavar=0.5, default=0.5, type=np.float, help = "minimal mass for picking (default 0.5)")
    
    # GPU
    parser.add_option("--gpu_id", metavar='1', default=None, type='string', help= "used GPU id number (None to use all devices)")
    
    # DEBUG
    parser.add_option("--debug", action='store_true', default=False, help = "debug mode (default False)")
 
    # advanced
    parser.add_option("--mode", metavar='pick', default=None, type='string', help = SUPPRESS_HELP )
    parser.add_option("--prob_thres", metavar=0.5, default = 0.5, type=np.float, help = SUPPRESS_HELP)
    parser.add_option("--adpmass", action='store_true', default=False, help = SUPPRESS_HELP)
    parser.add_option("--enhance", action='store_true', default=True, help = SUPPRESS_HELP)
    parser.add_option("--dust_removal", action='store_true', default=True, help = SUPPRESS_HELP)
    parser.add_option("--cvk_size", metavar = 7, default = 7, type=np.int, help = SUPPRESS_HELP)
    parser.add_option("--saturate_rate", metavar=0.1, default=0.1, type=np.float, help = SUPPRESS_HELP)
    parser.add_option("--repick", action='store_true', default=False, help = SUPPRESS_HELP)
    parser.add_option("--ext_format", metavar='star', default='star', type='string', help = SUPPRESS_HELP)
    
   
    (global_options, global_args) = parser.parse_args()
    command = ' '.join(sys.argv)
    
    global_mode = 'pick'
    
    print("================================================================================")
    print("==============      PARSED (PARticle SEgmentation Detector)       ==============")
    print("================================================================================\n")
    print("Choosing mode:\t%s" %global_mode)
    print("Initializing model framework")
    
    if global_mode == 'pick':

        # Loading third party packages
        import h5py
        import mrcfile
        import cv2

        # tracking particles
        import pandas as pd
        import trackpy as tp
        
        import mic_preprocess
        #import matplotlib.pyplot as plt
        
        if global_options.repick != True:
            if global_options.gpu_id != None:
                os.environ["CUDA_VISIBLE_DEVICES"] = global_options.gpu_id
            # keras/tf
            # limit tf memory usage
            import tensorflow.compat.v1 as tf  # SJHS
            tf.disable_v2_behavior()  # SJHS

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            #config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            session = tf.Session(config=config)

            from keras.utils import Sequence
            
            from keras import backend as K
            from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten , add, Conv2DTranspose , Reshape , Activation ,LeakyReLU , Multiply , Lambda
            from keras.models import Model, load_model
            from keras import optimizers

            class dataGenMicrograph():
                def __init__(self, name , gauFactor , mname , dataset_angPixel , inputSize = 1024 , reverseI = True , highPass = 500 , hard = True , lowerBound = 0 , upperBound = 3, DEBUG=False):

                    self.inputMap = np.zeros( (2 , inputSize , inputSize , 1) , dtype = np.float32)

                    with mrcfile.open(name , 'r' , permissive=True) as mrc:
                        origMap  = mrc.data
                        origMapNC= mic_preprocess.nmlizeC(iput = origMap , cscale = 4)
                    downScale , GauSigma , kSize = gauFactor
                    
                    # checkpoint mrcread
                    if global_timer:                        
                        timeseries_end1.append(time.time())
                    
                    filterMap = cv2.GaussianBlur( origMapNC , (kSize , kSize) , GauSigma) #/ 0.1 #ampCont

                    if reverseI == True:
                        filterMap = - filterMap

                    inner_ny , inner_nx , actualAngPixel_yx = mic_preprocess.downScaling(input_shape = origMapNC.shape, 
                                                                                         input_innerAngPixel = dataset_angPixel,
                                                                                         target_innerAngPixel = global_innerAngPixel)
                    downMap   = cv2.resize( filterMap , (inner_nx , inner_ny))

                    self.downMap = downMap

                    BPMaskS = mic_preprocess.genLPMaskS( inner_ny , inner_nx , np.mean(actualAngPixel_yx) * 2 , highPass , np.mean(actualAngPixel_yx) , 0 , 1)
                    BPMap   = mic_preprocess.LPFilterS( downMap , BPMaskS , 0 , 1)#Scale) #/ ny / nx * 4


                    dataFeeder_relu = mic_preprocess.nmlizeC(iput = BPMap , cscale = 4)

                    if DEBUG:
                        #plt.figure();plt.imshow(BPMap,'gray',vmin=0.)
                        print(np.mean(dataFeeder_relu),np.median(dataFeeder_relu),np.std(dataFeeder_relu))

                    if hard == False:
                        dataFeeder_relu_std = 1
                        dataFeeder_relu[dataFeeder_relu < (lowerBound * dataFeeder_relu_std)] = lowerBound * dataFeeder_relu_std
                        dataFeeder_relu[dataFeeder_relu > (upperBound * dataFeeder_relu_std)] = upperBound * dataFeeder_relu_std
                        dataFeeder_relu -= lowerBound * dataFeeder_relu_std
                        dataFeeder_relu /= (upperBound - lowerBound) * dataFeeder_relu_std
                        BPMap = dataFeeder_relu
                    else:
                        dataFeeder_relu_std = 1
                        dataFeeder_relu[dataFeeder_relu < (lowerBound * dataFeeder_relu_std)] = lowerBound * dataFeeder_relu_std
                        dataFeeder_relu[dataFeeder_relu > (upperBound * dataFeeder_relu_std)] = upperBound * dataFeeder_relu_std
                        dataFeeder_relu -= lowerBound * dataFeeder_relu_std
                        dataFeeder_relu /= (upperBound - lowerBound) * dataFeeder_relu_std
                        BPMap = dataFeeder_relu

                    print(inner_ny, inner_nx)
                    self.inputMap[0 , :inner_ny , :inner_nx , 0] += BPMap#downMap

                    # Recording originSize and downscaleSize
                    recordingSize(origMapNC.shape[0], origMapNC.shape[1] , inner_ny , inner_nx , name , mname = mname)
                    
                    # checkpoint preprocess
                    if global_timer:       
                        timeseries_end2.append(time.time())
            


            void_picking(global_options,global_args)
        
            session.close()
        else:
            void_picking(global_options,global_args)  
    
    else:
        print("Unrecognized mode choice !!")
    t_finish = time.time()
