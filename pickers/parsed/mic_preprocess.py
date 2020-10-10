import os
import sys
import glob
import time
from functools import partial
from optparse import OptionParser
from multiprocessing import Pool, Lock, Value

import cv2
import h5py
import numpy as np

import mrcfile


def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print("%s, %s start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("%s, %s end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("%.3fs taken for %s" % (time.time() - t0, func.__name__))
        return back
    return newFunc

def gauScaling(input_innerAngPixel = 1.3, lowThres = 30):
    # default lowThres value (30A) is enough
    AngPixel = input_innerAngPixel
    downScale = np.int(np.floor(lowThres / AngPixel / 2))
    GauSigma = lowThres * 3 / 2 / np.pi / AngPixel
    kSize = np.int(np.round(GauSigma * 4)) // 2 * 2 + 1
    return downScale , GauSigma , kSize

def downScaling(input_shape, input_innerAngPixel = 1.3, target_innerAngPixel = 15):
    input_ny = input_shape[0]; input_nx = input_shape[1]
    inner_ny = np.int(np.ceil(input_ny * input_innerAngPixel / target_innerAngPixel / 4 ) * 4)
    inner_nx = np.int(np.ceil(input_nx * input_innerAngPixel / target_innerAngPixel / 4 ) * 4)
    actualAngPixel_y = input_ny * input_innerAngPixel / inner_ny
    actualAngPixel_x = input_nx * input_innerAngPixel / inner_nx
    return inner_ny , inner_nx , (actualAngPixel_y , actualAngPixel_x)

def gauLPFilter(input_img , gauFactor):
    downScale , GauSigma , kSize = gauFactor
    output_img = cv2.GaussianBlur( input_img , (kSize , kSize) , GauSigma)
    return output_img

def readMRCfile(name):
    with mrcfile.open(name , 'r' , permissive=True) as mrc:
        origMap  = mrc.data
    return origMap
    
def nmlize(iput):
    iput=iput-np.mean(iput)
    iput=iput/np.std(iput)
    return iput

def nmlizeC(iput , cscale , cutsigma = -1):
    tsize = iput.shape
    if len(tsize) == 2:
        ty, tx = tsize
    elif len(tsize) ==3:
        _, ty, tx = tsize
        iput = iput[0]
    hty = ty // 2; htx = tx // 2
    cty = ty // cscale; ctx = tx // cscale
    iput=iput-np.mean(iput[ hty - cty : hty + cty , htx - ctx : htx + ctx ])
    iput=iput/np.std(iput[ hty - cty : hty + cty , htx - ctx : htx + ctx ])
    if cutsigma!=-1:
        csrange = iput[ hty - cty : hty + cty , htx - ctx : htx + ctx ].copy()
        cutmean = np.mean(csrange[np.abs(csrange)<cutsigma])
        cutstd = np.std(csrange[np.abs(csrange)<cutsigma])
        iput = iput - cutmean
        iput = iput / cutstd
    return iput

# Masked normalization
def nmlizeM(iput , mask , cscale , cutsigma = -1):
    tsize = iput.shape
    if len(tsize) == 2:
        ty, tx = tsize
    elif len(tsize) ==3:
        _, ty, tx = tsize
        iput = iput[0]
    mask = cv2.resize( mask.astype('uint8') , (iput.shape[::-1])).astype('bool')
    iput=iput-np.mean(iput[ ~mask ])
    iput=iput/np.std(iput[ ~mask ])
    if cutsigma!=-1:
        csrange = iput[ hty - cty : hty + cty , htx - ctx : htx + ctx ].copy()
        cutmean = np.mean(csrange[np.abs(csrange)<cutsigma])
        cutstd = np.std(csrange[np.abs(csrange)<cutsigma])
        iput = iput - cutmean
        iput = iput / cutstd
    return iput
    
def f2in(iput,mult):
    iput[iput>mult]=mult
    iput[iput<-mult]=-mult
    return np.round((iput+mult)*128./mult*256/257).astype('uint8')

def in2f(iput,mult):
    iput = iput.astype(np.float) / 128. * mult /256. * 257 - mult
    return iput

def convertINT(sample_data, sigmaN = 3 , bits=8 , scale = False):
    if scale == False:
        sample_data[sample_data >  sigmaN ] =  sigmaN
        sample_data[sample_data < -sigmaN ] = -sigmaN
    # aa   = np.linspace( -3 , 3 , 6 )
    # b    = ( aa + 3 ) / 6 * ( 2 ** 16 - 1 )
    # b_r  = np.round(( aa + 3 ) / 6 * ( 2 ** 16 - 1 ))
    # aa_o = b_r / ( 2 ** 16 - 1 ) * 6 - 3
        sample_data = np.round(( (sample_data + sigmaN) / sigmaN / 2 ) * (2 ** bits - 1)).astype('uint%d' %bits)
        return sample_data
    else:
        # default scale 0 - 128, or manually set one
        lowB = scale[0]; highB = scale[1]
        rangeB = scale[1] - scale[0]
        sample_data[sample_data > highB ] = highB
        sample_data[sample_data <  lowB ] =  lowB
        sample_data = sample_data - lowB
        sample_data = np.round(sample_data * (2 ** bits - 1) / rangeB).astype('uint%d' %bits)
        return sample_data
        

def revertINT(sample_data, sigmaN = 3 , bits=8 , scale = False):
    if scale == False:
        sample_data = sample_data / (2 ** bits) * sigmaN * 2 - sigmaN
        return sample_data
    else:
        # default scale 0 - 128
        lowB = scale[0]; highB = scale[1]
        rangeB = scale[1] - scale[0]
        sample_data = sample_data * rangeB / (2 ** bits -1 ) + lowB
        return 0

def gen_convKernel(Radius):
    xx2 , yy2 = np.meshgrid(np.linspace(0 , Radius * 2 , Radius * 2) , np.linspace(0 , Radius * 2 , Radius * 2))
    zz2 = ( xx2 - Radius) ** 2 + ( yy2 - Radius) ** 2
    return zz2 < Radius ** 2 , zz2

def DRAWcoor_filter(coorList , index , prefixLabel):
    indexDst = np.zeros( index.shape , dtype = np.float )
    indexDep = np.zeros( ( index.shape[0] + dRadius * 2 , index.shape[1] + dRadius * 2 ) , dtype = 'uint8' )
    coorNumList = []    
    coorNum = 0
    for coor in coorList:
        coorY = int(coor[0])
        coorX = int(coor[1])
        if indexDep[ coorY + dRadius , coorX + dRadius ]: 
            coorNum += 1
            continue
        if index[ coorY - mRadius : coorY + mRadius , coorX - mRadius : coorX + mRadius ].shape == ( mRadius*2 , mRadius*2 ):
            index[ coorY - mRadius : coorY + mRadius , coorX - mRadius : coorX + mRadius ] += circleMask
            if prefixLabel == '_SELECT':
                tRadius = np.int( np.sqrt( coorList[ coorNum , 2 ] / np.pi ) * 2 * T_factor )
                indexDep[ coorY : coorY + tRadius * 2 , coorX : coorX + tRadius * 2 ] += gen_convKernel( tRadius )[0]
            else:
                indexDep[ coorY : coorY + dRadius * 2 , coorX : coorX + dRadius * 2 ] += depressMask
            indexDst[ coorY - mRadius : coorY + mRadius , coorX - mRadius : coorX + mRadius ] += circleMask_dst
            coorNumList.append(coorNum)
            coorNum += 1
            continue
        coorNum += 1

    return index > 0 , indexDst , coorNumList


## http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
def DEFcont(contour):
    (x,y),radius = cv2.minEnclosingCircle(contour)
    center = (int(x),int(y))
    return cv2.contourArea(contour) , center[1] , center[0] #, cy ,cx 


def genLPMask( ny , nx , loPass , hiPass , angPix , LPtype):
    cy , cx = ny //2 , nx // 2
    loRadius_y = np.ceil( ny * angPix / loPass ).astype('int16');loRadius_x = np.ceil( nx * angPix / loPass ).astype('int16')
    hiRadius_y = np.ceil( ny * angPix / hiPass ).astype('int16');hiRadius_x = np.ceil( nx * angPix / hiPass ).astype('int16')
    loMask = genMask(loRadius_x , loRadius_y)[0];hiMask = genMask(hiRadius_x , hiRadius_y)[0]
    mask = np.zeros( ( ny , nx , 2), dtype = np.float )
    if LPtype == 0:
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,0] = loMask
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,1] = loMask
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,0] = 1 - hiMask
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,1] = 1 - hiMask
    elif LPtype == 1:
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,0] = hiMask
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,1] = hiMask
    else:
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,0] = loMask
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,1] = loMask
    return mask    

def genLPMaskS( ny , nx , loPass , hiPass , angPix , LPtype , Scale):
    ny = ny // Scale ; nx = nx // Scale ; angPix = angPix * Scale
    cy , cx = ny //2 , nx // 2
    loRadius_y = np.ceil( ny * angPix / loPass ).astype('int16');loRadius_x = np.ceil( nx * angPix / loPass ).astype('int16')
    hiRadius_y = np.ceil( ny * angPix / hiPass ).astype('int16');hiRadius_x = np.ceil( nx * angPix / hiPass ).astype('int16')
    loMask = genMask(loRadius_x , loRadius_y)[0];hiMask = genMask(hiRadius_x , hiRadius_y)[0]
    mask = np.zeros( ( ny , nx , 2), dtype = np.float )
    if LPtype == 0:
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,0] = loMask
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,1] = loMask
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,0] = 1 - hiMask
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,1] = 1 - hiMask
    elif LPtype == 1:
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,0] = hiMask
        mask[cy-hiRadius_y:cy+hiRadius_y,cx-hiRadius_x:cx+hiRadius_x,1] = hiMask
    else:
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,0] = loMask
        mask[cy-loRadius_y:cy+loRadius_y,cx-loRadius_x:cx+loRadius_x,1] = loMask
    return mask    

def genLPMaskSS( ny , nx , loPass , hiPass , angPix , LPtype , Scale):
    ny = ny // Scale ; nx = nx // Scale ; angPix = angPix * Scale
    cy , cx = ny //2 , nx // 2
    loRadius_y = np.ceil( ny * angPix / loPass ).astype('int16');loRadius_x = np.ceil( nx * angPix / loPass ).astype('int16')
    hiRadius_y = np.ceil( ny * angPix / hiPass ).astype('int16');hiRadius_x = np.ceil( nx * angPix / hiPass ).astype('int16')
    loMask = genMask(loRadius_x , loRadius_y)[0];hiMask = genMask(hiRadius_x , hiRadius_y)[0]
    mask = np.zeros( ( loRadius_y * 2 , loRadius_x * 2 , 2), dtype = np.float )
    hcy = loRadius_y ; hcx = loRadius_x 
    if LPtype == 0:
        mask[hcy-loRadius_y:hcy+loRadius_y,hcx-loRadius_x:hcx+loRadius_x,0] = loMask
        mask[hcy-loRadius_y:hcy+loRadius_y,hcx-loRadius_x:hcx+loRadius_x,1] = loMask
        mask[hcy-hiRadius_y:hcy+hiRadius_y,hcx-hiRadius_x:hcx+hiRadius_x,0] = 1 - hiMask
        mask[hcy-hiRadius_y:hcy+hiRadius_y,hcx-hiRadius_x:hcx+hiRadius_x,1] = 1 - hiMask
    elif LPtype == 1:
        mask[hcy-hiRadius_y:hcy+hiRadius_y,hcx-hiRadius_x:hcx+hiRadius_x,0] = hiMask
        mask[hcy-hiRadius_y:hcy+hiRadius_y,hcx-hiRadius_x:hcx+hiRadius_x,1] = hiMask
    elif LPtype == 3:
        mask[hcy-hiRadius_y:hcy+hiRadius_y,hcx-hiRadius_x:hcx+hiRadius_x,0] = 1 - hiMask
        mask[hcy-hiRadius_y:hcy+hiRadius_y,hcx-hiRadius_x:hcx+hiRadius_x,1] = 1 - hiMask        
    else:
        mask[hcy-loRadius_y:hcy+loRadius_y,hcx-loRadius_x:hcx+loRadius_x,0] = loMask
        mask[hcy-loRadius_y:hcy+loRadius_y,hcx-loRadius_x:hcx+loRadius_x,1] = loMask
    return mask    
    
def LPFilter(inputImg , LPMask , angPix):
    img = inputImg; mask = LPMask
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back0 = cv2.idft(f_ishift , flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
    return img_back0[ : , : , 0 ]

def LPFilterS(inputImg , LPMask , angPix , Scale):
    img = inputImg; mask = LPMask
    iy , ix = img.shape
    img = cv2.resize( img , ( ix // Scale , iy // Scale ) )
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back0 = cv2.idft(f_ishift , flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)

    return cv2.resize( img_back0[ : , : ,  ] , ( ix , iy ) )

def LPFilterSS(inputImg , LPMask , angPix , Scale):
    img = inputImg; mask = LPMask
    iy , ix = img.shape
    my , mx , _ = mask.shape
    cpy = (iy - my) // 2; cpx = (ix - mx) // 2
    #print((iy,ix),(my,mx),(cpy,cpx))
    img = cv2.resize( img , ( ix // Scale , iy // Scale ) )
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift[cpy : -cpy , cpx : -cpx] * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back0 = cv2.idft(f_ishift , flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)

    return cv2.resize( img_back0[ : , : ,  ] , ( ix , iy ) )

def genMask(xRadius , yRadius):
    xx2 , yy2 = np.meshgrid(np.linspace(0 , xRadius * 2 , xRadius * 2) , np.linspace(0 , yRadius * 2 , yRadius * 2))
    zz2 = ( xx2  / xRadius - 1. ) ** 2 + ( yy2 / yRadius - 1. ) ** 2
    return zz2 <= 1. , zz2
        
def input_type(pattern , serial = False):
    input_ = input('Input %s:\n' %pattern)
    while 1:
        try:
            if serial:
                output = input_
            else:
                output = float(input_);
            break
        except:
            input_ = input('Wrong input')
    return output;
    
def input_item():
    dsName = input_type('Dataset Name', serial = True)
    dsAngPixel = input_type('Angstrom in Pixel\t(AngPix)')
    dsAmpCont= input_type('Amplified contrast\t(AmpCont)')
    dsKV    = input_type('Voltage for EM\t(Kv)')
    dsCs    = input_type('Spherical Aberration\t(Cs)')
    dsMagn    = input_type('Detector Size in um\t(detector)')
    dsPATH    = input_type('PATH for dataset' , serial = True)
    dsPattern  = input_type('PATTERN for dataset' , serial = True)
    dsSuffix = input_type('SUFFIX for dataset' , serial = True)
    return dsName, dsAngPixel, dsAmpCont, dsKV , dsCs, dsMagn, dsPATH, dsPattern, dsSuffix

# https://stackoverflow.com/questions/2080660/python-multiprocessing-and-a-shared-counter
def pool_init(c,l):
    ''' store the counter for later use '''
    global global_lock, global_nameNum
    global_nameNum = c
    global_lock = l
