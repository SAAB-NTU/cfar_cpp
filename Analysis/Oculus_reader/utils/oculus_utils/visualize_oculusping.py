import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
from skimage.util import img_as_float
from skimage.transform import resize, rotate
from scipy.ndimage import map_coordinates
import scipy
from scipy import ndimage, misc
import cv2
import math
import itertools
import time
import json

#sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#from tools.fls.read_oculusping import OculusFLSPing, OculusPing_generator
from read_oculusping import OculusFLSPing, OculusPing_generator

binfile_path = './data/raw/FLS_Stream/Oculus_20200304_161016.bin' #'../../data/raw/FLS_Stream/Oculus_20200304_161016.bin'
'''
save_dir = '../Saab_dataset_SonarImages/Set1-Test_Site_B+02Oct2019+13h00m49s+3_4gamma-6intensity/'
save_dir_ranges = '../Saab_dataset_SonarImages/Set1-Test_Site_B-w_ranges/'
timestamp_filename = 'Frame_Timestamps_Site_B.csv'
pingdict_jsonfile = 'Ping_Dict-Site_B.json'
'''

opencv_colormaps = {'autumn': cv2.COLORMAP_AUTUMN,
                        'bone': cv2.COLORMAP_BONE,
                        'jet': cv2.COLORMAP_JET,
                        'winter': cv2.COLORMAP_WINTER,
                        'rainbow': cv2.COLORMAP_RAINBOW,
                        'ocean': cv2.COLORMAP_OCEAN,
                        'summer': cv2.COLORMAP_SUMMER,
                        'spring': cv2.COLORMAP_SPRING,
                        'cool': cv2.COLORMAP_COOL,
                        'hsv': cv2.COLORMAP_HSV,
                        'pink': cv2.COLORMAP_PINK,
                        'hot': cv2.COLORMAP_HOT,
                        'parula': cv2.COLORMAP_PARULA,
                        'magma': cv2.COLORMAP_MAGMA ,
                        'inferno': cv2.COLORMAP_INFERNO,
                        'plasma': cv2.COLORMAP_PLASMA ,
                        'viridis': cv2.COLORMAP_VIRIDIS,
                        'cividis': cv2.COLORMAP_CIVIDIS,
                        'twilight': cv2.COLORMAP_TWILIGHT,
                        'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED,
                        'turbo': cv2.COLORMAP_TURBO}

def Deg2Rad(degrees):
    return (degrees/180.0) * np.pi

def Rad2Deg(radians):
    return (radians/np.pi) * 180.0

def show_image(img, title, mapping="gray", blocking=True):
    '''
    Function to show an image using the Pyplot/PLT library into a Figure

    Arguments:
    - img => Input image to be displayed
    - mapping => Type of image mapping used (the plt.imshow's cmap variable)
    - title => Title of the figure to be created
    - blocking => Argument on whether the displaying of the figure should be process blocking or not.
    '''
    plt.figure()
    plt.imshow(img, cmap=mapping)
    plt.title(title)
    plt.show(block=blocking)

def check_for_file(file_path, create_if_not=False):
    '''
    Simple function to check availability of specified file.
    Also could be used to create specified file if flag is triggered as True

    Arguments:
    - file_path => Path of file to be checked
    - create_if_not => Flag to indicate whether to create the specified file if it's currently not available

    Returns:
      None
    '''
    file_exists = os.path.isfile(file_path)

    if file_exists:
        return True
    else:
        if create_if_not:
            return False


def calc_physical_ping_dims(im_H, im_W, max_physRange, opening_angle):
    '''
    Function to calculate the physical height, width and length reprsentation of sonar image's Height, Width, and pixels, respectively.

    Arguments:
    - im_H => Sonar Image's pixel Height
    - im_W => sonar Image's pixel Width
    - max_physRange => Maximum Physical range of that is able to be shown by the sonar.
    - opening_angle => Opening viewing angle of current sonar ping object.

    Returns:
    - phys_Height => Physical range (in m) represented by image's Height
    - phys_Width => Physical range (in m) represented by image's Width
    - lenofPixel => Physical range (in m) represented by each of image's pixels.
    '''
    
    '''
    vp_left = -0.9 * max_physRange
    vp_right = 0.9 * max_physRange

    vp_bottom = -0.02 * max_physRange
    vp_top = 1.02 * max_physRange

    lenofPixel_x = np.abs(vp_right - vp_left) / im_W
    lenofPixel_y = np.abs(vp_top - vp_bottom) / im_H
    '''
    maxRange = max_physRange

    max_angle = math.asin(float(im_W)/(2.*im_H)) #arcsinus
    if max_angle >= (opening_angle / 2.0):
        print("Phys dims case A", max_angle, opening_angle)
        phys_Height = maxRange
        lenofPixel = phys_Height / im_H
        phys_Width = im_W * lenofPixel
        
    else:
        print("Phys dims case B", max_angle, opening_angle)
        phys_Width = 2 * maxRange * math.sin(opening_angle / 2.0)
        lenofPixel = phys_Width / im_W
        phys_Height = im_H * lenofPixel
    
    return phys_Height, phys_Width, lenofPixel



def calcCartesian(row, col, phys_H, phys_W, pix_len):
    '''
    Function to translate row, col indexes into physical x,y cartesian coordinates

    Arguments:
    - row => row index. Could be of singular integer row_id, or matrix row_id-s
    - col => column index. Could be of singular integer column_id, or matrix of column_id-s
    - phys_H => Calculated physical range (in meters) value represented by image's Height
    - phys_W => Calculated physical range (in meters) value represented by image's Width
    - pix_len => physical range (in meters) represented by a pixel

    Returns:
    - Singular x, y coordinate pairs if provided row and col id-s are of singular integers, or
    - x, y coordinate matrix if provided row and col id-s are of index matrices instead. 
    '''
    x = phys_H - (pix_len * (row + 0.5))
    y = pix_len * (col + 0.5) - (phys_W / 2.0)
    #print("Cartesian for (row)(col)", row, col, "| (x, y): ", x, y)
    return x, y 


#TODO: Try out not using the interpolation method (which was mainly used for the Saab dataset anyway). This is considering that the 
# data for Oculus is already processed mainly by the sonar hardware, so not using interpolation like this might yield better results overall
def get_MatrixPixelValues(ping_config, brgTable, x_m, y_m, echoVals, viewer_settings, verbose=False):
    '''
    Calculates the appropriate floating point value intensities for each pixel of the sonar image, 
    based on the physical x and y coordinates associated for each pixels and their associated echo data
    (also floating point values obtained based on sonar ping's data).

    Note that (in accordance with Saab's Playback code) each pixel's value is determined using a 
    Bilinear Interpolation  operation, described below:

    Variables for interpolation:
		theta1 = angle of "left" beam
		theta2 = angle of "right" beam
		theta  = requested angle

		r1 = range for "lower" samples
		r2 = range for "upper" samples
		polar_r = requested range

		response11 = response from point 11: p(1,1)
		response12 = response from point 12: p(1,2)
		response22 = response from point 22: p(2,2)
		response21 = response from point 21: p(2,1)

    Visualization of pixel value:

       p(1,2) |         | p(2,2)
	  r2  ----o---------o---
			  | (x,y)   |				o : sample value
			  | +       |				+ : pixel coordinate
			  |         |
	  r1  ----o---------o---
	   p(1,1) |         | p(2,1)

			theta1       theta2
    
    Arguments:
    - ping_settings => Collection of settings/parameters for current sonar ping
    - x_m => Matrix representing the Physical x coordinate values associated for sonar image's pixels
    - y_m => Matrix representing the Physical y coordinate values associated for sonar image's pixels
    - echoVals => Matrix representing the echo values associated with each sonar image's pixels
    - viewer_settings => Collection of settings chosen by user to view the Sonar's Ping image 

    Returns:
    - pix_vals => Matrix representing calculated Floating point Intensity values for each of sonar image's pixels
    - pix_bool => Mask Boolean Matrix, indicating which pixels are actually part of sonar image (True), or part of Sonar Viewer's background (False)
    '''
    
    #Prepare the necessary scalar parameters
    
    M = ping_config['M']
    N = ping_config['N']
    minRange = 0 #ping_settings['start_range']
    maxRange = M * ping_config['range_resolution'] #ping_config['user-stop_range']
    #opening_angle = ping_config['swath_open']
    gamma_correction = ping_config['user-gamma_correction']

    
    
    opening_angle = Deg2Rad(brgTable[-1] - brgTable[0])

    #In Saab's playback GUI, part of parameters that are set/chosen by user. 
    brightness_coeff = viewer_settings['brightness_coeff']

    #Corrected echo values of pixels, subject to viewer's choice of brightness and gamma_correction coefficients
    #echoData = np.minimum(1.0, brightness_coeff * echoVals * np.power(echoVals, 1/gamma_correction))
    echoData = echoVals

    #preparation of pixel values matrix (Not completely necessary, feel free to delete.)
    pix_vals = np.zeros(x_m.shape, dtype=np.float64) #np.zeros(x_m.shape, dtype=np.float32)

    #Matrix contanining the r-component of each pixel's polar coordinate representation
    polar_r = np.sqrt(np.power(x_m, 2) + np.power(y_m, 2))
    print("Maximum range", maxRange, M * ping_config['range_resolution'])

    #Calculate the "upper" and "lower" range and r-component value of physical points separated by each pixel 
    # (Remember that each pixels indicate a 'distance' separation between 2 actual physical points/ranges)
    sample_incr = (maxRange - minRange) / (M)# - 1)
    rangeId1 = np.floor( (polar_r - minRange) / sample_incr ).astype(np.int16)
    rangeId2 = np.ceil( (polar_r - minRange) / sample_incr ).astype(np.int16)
    r1 = minRange + np.multiply(rangeId1, sample_incr) #range for lower samples
    r2 = minRange + np.multiply(rangeId2, sample_incr) #range for upper samples

    rangeId = np.round((polar_r - minRange)/sample_incr).astype(np.int16)
    r = minRange + np.multiply(rangeId, sample_incr)
    
    #Calculate the theta-component of pixel's polar coordinate representation
    theta = np.arctan2(y_m, x_m)
    #np.savetxt('theta-test.csv', theta, delimiter=',')

    # 'Upper' and 'lower' channel id-s for each pixel.
    channelId1 = np.zeros(theta.shape, dtype=rangeId1.dtype)
    channelId2 = np.zeros(theta.shape, dtype=rangeId2.dtype)

    channelId = np.zeros(theta.shape, dtype=rangeId.dtype)

    #Calculate angle parameters of the sonar image. 
    #Note that startAngle and endAngle indicates the start and end angles from which valid sonar image pixels are viewable
    upperBound = opening_angle / 2
    lowerBound = -upperBound
    angleStep = opening_angle/ N #use the brgTable instead(?)
    startAngle = lowerBound + (angleStep / 2)
    endAngle = upperBound - (angleStep / 2)
    
    print("Angles parameters", startAngle, endAngle)

    #Debugging purposes
    if verbose:
        print("Angles", lowerBound, upperBound, startAngle, endAngle)
        print("Angles (degrees)", np.degrees(lowerBound), np.degrees(upperBound), np.degrees(startAngle), np.degrees(endAngle), np.degrees(opening_angle))

    response = np.zeros(pix_vals.shape, dtype=echoData.dtype)
    
    #Prepare the matrices for each pixel's bilinear interpolation variables
    theta1 = np.zeros(pix_vals.shape, dtype=theta.dtype)
    theta2 = np.zeros(pix_vals.shape, dtype=theta.dtype)
    response11 = np.zeros(pix_vals.shape, dtype=echoData.dtype)
    response21 = np.zeros(pix_vals.shape, dtype=echoData.dtype)
    response22 = np.zeros(pix_vals.shape, dtype=echoData.dtype)
    response12 = np.zeros(pix_vals.shape, dtype=echoData.dtype)

    #Case for pixels whose theta values are outside of boundary values.
    case0 = (theta < lowerBound) | (theta > upperBound)
    case_extra = ((rangeId1 >= 0) & (rangeId2 < M))

    #cases of pixels that have angle orientation that are less than starting Angle
    case1 = theta < startAngle
    theta1[case1] = lowerBound
    theta2[case1] = startAngle
    response11[case1] = 0
    response21[case1 & case_extra] = echoData[rangeId1[case1 & case_extra] * N]
    response22[case1 & case_extra] = echoData[rangeId2[case1 & case_extra] * N]
    response12[case1] = 0

    plt.figure()
    plt.subplot(221)
    plt.imshow(response11, cmap='hot')
    plt.subplot(222)
    plt.imshow(response21, cmap='hot')
    plt.subplot(223)
    plt.imshow(response22, cmap='hot')
    plt.subplot(224)
    plt.imshow(response12, cmap='hot')
    plt.title("responses for case 1")
    plt.show(block=False)

    #Case of pixels with angle orientation that are more than end Angle
    case2 = theta > endAngle
    theta1[case2] = endAngle
    theta2[case2] = upperBound
    response11[case2 & case_extra] = echoData[(rangeId1[case2 & case_extra] + 1) * N - 1]
    response21[case2] = 0
    response22[case2] = 0
    response12[case2 & case_extra] = echoData[(rangeId2[case2 & case_extra] + 1) * N - 1]

    plt.figure()
    plt.subplot(221)
    plt.imshow(response11, cmap='hot')
    plt.subplot(222)
    plt.imshow(response21, cmap='hot')
    plt.subplot(223)
    plt.imshow(response22, cmap='hot')
    plt.subplot(224)
    plt.imshow(response12, cmap='hot')
    plt.title("responses for case 2")
    plt.show(block=False)

    #Case of pixels whose angle orientation are appropriately between starting & end angle values
    case3 = ~(case0 | case1 | case2)  &  case_extra #Equivalent to (~case0 & ~case1 & ~case2) [deMorgan's Law]
    channelId1[case3] = np.floor((theta[case3] - startAngle) / angleStep)
    channelId2[case3] = np.ceil((theta[case3]- startAngle) / angleStep)
    theta1[case3] = startAngle + (channelId1[case3] * angleStep)
    theta2[case3] = startAngle + (channelId2[case3] * angleStep)
    response11[case3] = echoData[rangeId1[case3] * N + channelId1[case3]]
    response21[case3] = echoData[rangeId1[case3] * N + channelId2[case3]]
    response22[case3] = echoData[rangeId2[case3] * N + channelId2[case3]]
    response12[case3] = echoData[rangeId2[case3] * N + channelId1[case3]]
    #np.savetxt('channelId_example.csv', channelId1, delimiter=',')
    #np.savetxt('rangeId_example.csv', rangeId1, delimiter=',')
    print("number of acoustic image pixels", echoData.shape, M * N)
    print(np.min(rangeId1[case3] * N + channelId1[case3]), np.max(rangeId1[case3] * N + channelId1[case3]))
    print("Max two", np.max(polar_r[case3]), response11[case3].shape)

    plt.figure()
    plt.subplot(221)
    plt.imshow(response11, cmap='hot')
    plt.subplot(222)
    plt.imshow(response21, cmap='hot')
    plt.subplot(223)
    plt.imshow(response22, cmap='hot')
    plt.subplot(224)
    plt.imshow(response12, cmap='hot')
    plt.title("responses for case 3")
    plt.show(block=False)

    rangeAngleDiff = ((r2 - r1) * (theta2 - theta1))

    pix_vals = (response11 * (r2 - polar_r) * (theta2 - theta) / rangeAngleDiff) + \
            (response12 * (polar_r - r1) * (theta2 - theta) / rangeAngleDiff) + \
            (response21 * (r2 - polar_r) * (theta - theta1) / rangeAngleDiff) + \
            (response22 * (polar_r - r1) * (theta - theta1) / rangeAngleDiff) 
    
    #Calculate the pix_bool mask indicating which pixels are part of sonar ping, or not
    pix_bool = np.ones(pix_vals.shape, dtype='bool')
    pix_bool[(rangeId1 < 0) | (rangeId2 >= M)] = False
    pix_bool[case0] = False #[(theta < lowerBound) | (theta > upperBound)] = False
    pix_bool[(channelId1 < 0) | (channelId2 >= N)] = False

    '''
    case0 = (theta < lowerBound) | (theta > upperBound)
    case1 = ~(case0)

    response[case0] = 0
    channelId = np.round( (theta - startAngle) / angleStep).astype(np.int16)
    np.savetxt('ChannelId_test.csv', channelId, delimiter=',')
    response[case1] = echoData[rangeId[case1] * N + channelId[case1]]
    #print("Size of case 1:", response[case1].shape)
    show_image(np.reshape(echoData, (M, N)), "echoData before processing", mapping='hot', blocking=False)
    show_image(response, "response after process", mapping='hot', blocking=False)
    rangeAngleDiff = r * theta #?

    pix_vals = response 

    pix_bool = np.ones(pix_vals.shape, dtype='bool')
    pix_bool[(rangeId < 0) | (rangeId >= M)] = False
    pix_bool[case0] = False
    pix_bool[(channelId < 0) | (channelId >=N)] = False
    '''
    #Case where there are pixels with 0 range Angle Difference 
    # (not good as this means that this pixel will likely overlap its neighbors in terms of angular orientation.)
    if 0.0 in rangeAngleDiff[:]:
        print("There are coordinates where Range Angle diff is equals Zero!! Pixel values: ", pix_vals)
    print(pix_vals.dtype, np.max(pix_vals))
    return pix_vals, pix_bool

#applies an orthographic projection operation, similar to the ones shown in QMatrix's ortho function:
#https://doc.qt.io/archives/qt-4.8/qmatrix4x4.html#ortho
def apply_ortho(cur_m, vp_left, vp_right, vp_bottom, vp_top, near=-1.0, far=1.0):
    ortho_matrix = np.zeros()
    return

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def fan_project(echo_img, brgTable_rad, img_rows, img_cols):
    """
    Helper function to project image from a normal rectangular echo image to a fan-display image
    """
    fan_img_rows, fan_img_cols = img_rows, int(np.abs(2 * img_rows * np.sin(brgTable_rad[-1])))
    fan_img = np.zeros((fan_img_rows, fan_img_cols), dtype=echo_img.dtype)

    #mapping process starts here
    brg_sin, brg_cos = np.sin(brgTable_rad), np.cos(brgTable_rad)

    #round indexes up or down?
    for i in range(0, int(fan_img_rows)): 
        col_idx = np.ceil((fan_img_cols/2) + (i * brg_sin)).astype(np.int16)     
        row_idx = np.ceil(i * brg_cos).astype(np.int16)      
        np.putmask(row_idx, row_idx>=fan_img_rows, fan_img_rows-1)
        np.putmask(col_idx, col_idx>=fan_img_cols, fan_img_cols-1)
        fan_img[row_idx, col_idx] = echo_img[i, :]

    return fan_img, fan_img_rows, fan_img_cols

def OculusPing2Img(oculus_ping, usr_imHeight=None, usr_imWidth=None, use_scipy_remap=True, spline_order=1, viewer_settings=None, verbose=False):
    '''
    Function to convert each Oculus ping's data into a sonar image.

    Arguments:
    - oculus_ping => an object representing a sonar's Ping
    - usr_imHeight => Specification for the final produced sonar image's Height
    - usr_imWidth => Specification for the final produced sonar image's Width
    - use_scipy_remap => Flag to indicate whether to use Scipy's remap/map_coordinate function or use 
                         the fan_project function for fan remapping. Default value is True
    - spline_order => Value of spline interpolation order to be used in the remapping (especially if using scipy's remap).
                      Default value is 1
    - viewer_settings => (Optional) A dictionary collection of settings specified by user/viewer.
                         If None are provided, function reverts to default values, specified below. 
    - verbose => (Optional) Flag to indicate whether function should provide verbose messages
    
    Returns:
    - new_img => unsigned 8-bit Numpy Image array representing the inputted oculus_ping.
    - polar2cart_x => polar to cartesian x values if using cv remap. 
    - polar2cart_y => polar to cartesian y values if using cv remap.
    '''

    #check whether user issued a new set of data settings. If not, use default values
    if viewer_settings is None:
        viewer_settings = {'brightness_coeff': 5.0,
                            'gamma_correction': 3.4 ,
                            'lowlimit_Intensity': 0.2,
                            'maxlimit_Intensity': 6.0}

    pingConfig = oculus_ping.get_pingConfig()
    brgTable = oculus_ping.get_brgTable_val()

    M = pingConfig['M'] #nRanges
    N = pingConfig['N'] #nBeams

    opening_angle = brgTable[-1] - brgTable[0]
    opening_angle_rad = np.radians(opening_angle)#opening_angle_rad = Deg2Rad(opening_angle)
    if verbose:
        print("Opening angle degree, rad", opening_angle, opening_angle_rad)


    #Obtain the ping's echo data, subject to the Intensity's maximum limit specified
    #(In original code, this seems to not be an option available to user, but I'm making 
    # this version modifiable via viewer_settings variable)
    #echoData = oculus_ping.get_echoData(maxlimit_Intensity, verbose=True)
    echoData = oculus_ping.get_acousticImg_pix()
    if len(echoData) == 0:
        print("No water column pixels detected in this ping. Something is wrong with this ping data!")
        return False, None, None

    #debugging purposes
    if verbose:
        print("Shape of echo data", echoData.shape)
    
    raw_img = oculus_ping.get_acousticImg_matrix()
    polar2cart_x, polar2cart_y = None, None #remapping functions that are utilized, in case user wants it

    if use_scipy_remap:
        """
        #Remapping function based on inverse warp_polar implementation found in:
        #https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/2
        """
        #determine cartesian dimensions of new image
        r = raw_img.shape[0]
        new_img = np.zeros((r * 2, r * 2), dtype=raw_img.dtype)
        cart_center = np.array(new_img.shape) / 2 - 0.5 #0.5 to eliminate un-even numbers
        out_h, out_w = new_img.shape
        
        #NOTES:
        # - mgrid is the shorter variation of meshgrid.
        # - mgrid produces shape of (2, out_h, out_w), hence the need to reshape o from (2,) to (2, 1, 1) via o[:, None, None]
        # - ys, xs are Cartesian coordinates of new_img's pixels that are centered. 
        ys, xs = np.mgrid[:out_h, :out_w] - cart_center[:,None,None] 

        #Determine remapping function for new image, i.e.
        #rs = 'r' coordinate of reference 'raw_img' pixel to be mapped onto (xs, ys) in 'new_img'
        #ts = 't' coordinate of reference 'raw_simg' pixel to be mapped onto (xs, ys) in 'new_img'
        rs = (ys**2+xs**2)**0.5
        ts = -np.arctan2(ys, xs) #Negative value needed as Scipy seems to have an 'inverted' column direction vs. Numpy  #np.arccos(xs/rs)

        #ts[ys<0] = - (opening_angle_rad - ts[ys<0]) #np.pi*2 - ts[ys<0] #adjustment mapping for negative theta. Not necessary in our case 
        
        #scale the theta value against raw_img's opening angle value (in rad) 
        ts *= (raw_img.shape[1]-1)/opening_angle_rad  

        #Remap using scipy's map_coordinate function
        map_coordinates(raw_img, (rs, ts), output=new_img, order=spline_order)

        #Rotation & slicing needed as map_coordinate's 0 theta starts at 3 o'clock position.
        new_img = rotate(new_img, -(90 + (opening_angle/2) ))
        new_img = new_img[r:, r - int(r * np.sin(opening_angle_rad / 2)):r + int(r * np.sin(opening_angle_rad/2))]

        #Sanity check to see whether user requested a specific Height/Width value for new image. 
        # If there aren't any requests, then just return as is
        if (usr_imHeight is not None) and (usr_imWidth is not None):
            new_img = skimage.transform.resize(new_img, (usr_imHeight, usr_imWidth))

        polar2cart_x, polar2cart_y = rs, ts
        
    
    else:
        remapped_img, _, _ = fan_project(raw_img, np.radians(brgTable), M, N)

        #TODO: bilinear interpolation HAS NOT worked yet!
        x_space, y_space = np.linspace(0, remapped_img.shape[0]-1, remapped_img.shape[0]), np.linspace(0, remapped_img.shape[1] - 1, remapped_img.shape[1])
        x_coor, y_coor = np.meshgrid(x_space, y_space, indexing='ij')
        remapped_img = bilinear_interpolate(remapped_img, y_coor, x_coor)

        new_img = skimage.transform.resize(remapped_img, (usr_imHeight, usr_imWidth))
        polar2cart_x, polar2cart_y = None, None

    if verbose:
        print("Image dims ratio:", new_img.shape[0]/ new_img.shape[1]) #debugging to know ratio of current temporary image (We'd like to keep the image's ratio intact as much as possible)
        print()

    return new_img, polar2cart_x, polar2cart_y # coor_x_new, coor_y_new


def OculusPing_viewer(oculus_ping, im_H=480, im_W=640, viewer_settings=None, mapping='hot', view_dist=False, blocking=True, return_rgb=True, view_img=False, verbose=False):
    '''
    Function to view the 2D Acoustic Image representation for an Oculus Ping. Note that the default settings for the Image's Height and Width 
    (i.e. im_H & im_W) are based on the settings used by the original Oculus Viewer tool for viewing of the Oculus Ping as well (see the SonarSurface.cpp's
    Render function for more details)
    '''
    if viewer_settings is None:
        viewer_settings = {'brightness_coeff': 4.6,
                            'gamma_correction': 3.4 ,
                            'lowlimit_Intensity': 0.17,
                            'maxlimit_Intensity': 6.0}
    '''
    opencv_colormaps = {'autumn': cv2.COLORMAP_AUTUMN,
                        'bone': cv2.COLORMAP_BONE,
                        'jet': cv2.COLORMAP_JET,
                        'winter': cv2.COLORMAP_WINTER,
                        'rainbow': cv2.COLORMAP_RAINBOW,
                        'ocean': cv2.COLORMAP_OCEAN,
                        'summer': cv2.COLORMAP_SUMMER,
                        'spring': cv2.COLORMAP_SPRING,
                        'cool': cv2.COLORMAP_COOL,
                        'hsv': cv2.COLORMAP_HSV,
                        'pink': cv2.COLORMAP_PINK,
                        'hot': cv2.COLORMAP_HOT,
                        'parula': cv2.COLORMAP_PARULA,
                        'magma': cv2.COLORMAP_MAGMA ,
                        'inferno': cv2.COLORMAP_INFERNO,
                        'plasma': cv2.COLORMAP_PLASMA ,
                        'viridis': cv2.COLORMAP_VIRIDIS,
                        'cividis': cv2.COLORMAP_CIVIDIS,
                        'twilight': cv2.COLORMAP_TWILIGHT,
                        'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED,
                        'turbo': cv2.COLORMAP_TURBO}
    '''
    #ping_img, coor_x, coor_y = OculusPing2Img(oculus_ping, im_H, im_W, viewer_settings=viewer_settings, verbose=verbose)
    ping_img, coor_x, coor_y = OculusPing2Img(oculus_ping, spline_order=1)
    brgTable = oculus_ping.get_brgTable_val()
    ping_timestamp = oculus_ping.get_pingConfig()['pingStart_time']
    ping_img_original = False
    
    if (ping_img is False): 
        if verbose:
            print("Ping image cannot be created. Something is wrong with this ping data.")
        pass
        
    else: 
        col_coor = np.round(coor_x[:, 0], decimals=1)
        row_coor = np.round(coor_y[0, :], decimals=1)
        ping_img_original = np.copy(ping_img)
        ping_img = cv2.applyColorMap(ping_img.astype(np.uint8), opencv_colormaps[mapping])
        if return_rgb:
            ping_img = cv2.cvtColor(ping_img, cv2.COLOR_BGR2RGB)
        
        print("Final size of ping image:", np.shape(ping_img_original))
        
        if view_img:
            cv2.imshow("Sonar Ping stream", ping_img)#ping_img.astype(np.uint8))

        #If Cartesian distances need to be overlayed onto the image  
        if view_dist:

            #Mark the x and y Cartesian distances represented by each pixel's x-y coordinate.
            ping_img_range = np.copy(ping_img)
            ping_img_range = cv2.line(ping_img_range, (0,200), (400,600), (255, 255, 255), 1)
            ping_img_range = cv2.line(ping_img_range, (400, 600), (800, 200), (255, 255, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            for y in range(0, len(col_coor)):
                if col_coor[y] % 1 == 0:
                    ping_img_range = cv2.putText(ping_img_range, str(col_coor[y]), (5, y+2), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    ping_img_range = cv2.line(ping_img_range, (0, y), (5, y), (255, 255, 255) ,1)
            for x in range(0, len(row_coor)):
                if row_coor[x] % 1 == 0:
                    ping_img_range = cv2.putText(ping_img_range, str(row_coor[x]), (x-6, 592), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    ping_img_range = cv2.line(ping_img_range, (x, 600), (x, 595), (255, 255, 255) ,1)
            
            cv2.imshow("Sonar Ping with distances", ping_img_range.astype(np.uint8))

    if not blocking:
        print("not blocking")
        print()        
        cv2.waitKey(1)
        return ping_img.astype(np.uint8)
    else:
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        return ping_img_original

    
    


def stream_ping_continuous(path, im_H, im_W, n=None, start_view_number=0, viewer_settings=None, mapping='jet', no_ping_limit=100, save_dir=None, \
                         view_dist=False, name_template='Ping_', save_timestamp=False, save_ping_settings=False, verbose=False):
    '''
    Function to continuously read and visualize the sonar images from captured binary sonar stream, for n number of sonar pings. Also returns
    a collection of appended SonarPing objects if called as assignment to a variable. Loop is similar to the read_pings function.

    Note that if no n number is provided, will continously read until all pings are exhausted.

    Arguments:
    - path => File path to binary file from which sonar pings to be extracted from
    - im_H => Height of image stream to be displayed
    - im_w => Width of image stream to be displayed
    - n => (Optional) Number of sonar pings to be read from path. If None is given, will proceed to read in all pings at once.
    - start_view_number => (Optional) Id of sonar ping to start the viewing. Default value is 0, meaning viewing starts from beginning till end
    - viewer_settings => (Optional) A dictionary collection of settings specified by user/viewer. If None are provided, function reverts to default values, \
                         specified below. 
    - mapping => (Optional) Color mapping to be used to visualize the sonar images. If None are provided, default is to use OpenCV's 'Jet' colormapping
    - no_ping_limit => (Optional) Indicates how many 'invalid sonar ping'-s can be tolerated. Once there are more than specified number of consecutive sonar \
                        pings are encountered, viewer will exit loop and stop. Default value is 100. 
    - save_dir => (Optional) Directory to save the continuous stream of images to. If None are provided, doesn't save and just continuously views the image stream.
    - view_dist => (Optional) If True and save_dir is provided, saves Sonar Ping images overlayed with x-y Cartesian distance (in metres) markers into a \
                    separate variation of the save_dir directory.
    - name_template => (Optional) Template name to be used to label each of the generated ping images.
    - save_timestamp => (Optional) If save_dir is provided, indicates whether to save the timestamp of each ping into a csv file in save_dir or not.
    - save_ping_settings => (Optional) If save_dir is provided, indicates whether to save each recorded ping's settings into a .json file within the save_dir

    Returns:
    - collected_oculus_pings => List of SonarPing objects that were collected from the binary file recording of specified path.
    '''

    collected_oculus_pings = []
    ping_counter = 0
    prev_is_ping_header = False
    prev_packet_data = None 
    packet_timestamp = -1
    timestamp_data = { 'ping_name':[],
                       'packet_timestamp': [],
                       'ping_timestamp': []}
    no_ping_counter = 0
    no_ping_limit = no_ping_limit
    ping_dict = {}

    if viewer_settings is None:
        viewer_settings = {'brightness_coeff': 4.6,
                            'gamma_correction': 3.4 ,
                            'lowlimit_Intensity': 0.17,
                            'maxlimit_Intensity': 6.0}

    opencv_colormaps = {'autumn': cv2.COLORMAP_AUTUMN,
                        'bone': cv2.COLORMAP_BONE,
                        'jet': cv2.COLORMAP_JET,
                        'winter': cv2.COLORMAP_WINTER,
                        'rainbow': cv2.COLORMAP_RAINBOW,
                        'ocean': cv2.COLORMAP_OCEAN,
                        'summer': cv2.COLORMAP_SUMMER,
                        'spring': cv2.COLORMAP_SPRING,
                        'cool': cv2.COLORMAP_COOL,
                        'hsv': cv2.COLORMAP_HSV,
                        'pink': cv2.COLORMAP_PINK,
                        'hot': cv2.COLORMAP_HOT,
                        'parula': cv2.COLORMAP_PARULA,
                        'magma': cv2.COLORMAP_MAGMA ,
                        'inferno': cv2.COLORMAP_INFERNO,
                        'plasma': cv2.COLORMAP_PLASMA ,
                        'viridis': cv2.COLORMAP_VIRIDIS,
                        'cividis': cv2.COLORMAP_CIVIDIS,
                        'twilight': cv2.COLORMAP_TWILIGHT,
                        'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED,
                        'turbo': cv2.COLORMAP_TURBO}

    with open(path, "rb") as bin_file:
        portnum_bytes = bin_file.read(2)
        port_number = int.from_bytes(portnum_bytes, byteorder='big', signed=False)
        
        packet_range = None
        if n is not None:
            packet_range = range(0, n)
        else:
            packet_range = itertools.count()

        for i in packet_range:
            #Case where last packet from previous loop iteration is actually start of a new ping.
            if prev_packet_data != None and prev_is_ping_header:
                #print("Creating ping data from previous unused packet")
                if verbose:
                    print("Collecting data for ping #{} from previous unused packet".format(str(ping_counter)))
                oculus_ping = init_oculus_ping(prev_packet_data)
                packet_timestamp = prev_packet_timestamp
                if oculus_ping == None:
                    print("Failed to create sonar ping object!")
                    sys.exit(1)
            else:
                packet_timestamp, _, packet_data, preamble = parse_packet_info(bin_file, verbose=verbose)
                if preamble:
                    if verbose:
                        print("Found valid ping preamble. Collecting data for ping #:", ping_counter)
                    oculus_ping = init_oculus_ping(packet_data)
                    no_ping_counter = 0
                    if oculus_ping == None:
                        print("Failed to create sonar ping object!")
                        sys.exit(1)
                elif not prev_is_ping_header :
                    if verbose:
                        print("Noise ping data detected. Skipping...")
                    oculus_ping = None
                    no_ping_counter += 1
                    if no_ping_counter == no_ping_limit:
                        break
                    continue

            #Collect and save the valid data bytes into a single sonar ping.
            prev_is_ping_header, prev_packet_data, prev_packet_timestamp = ping_capture_helper(oculus_ping, bin_file, 0, 0)
            collected_oculus_pings.append(oculus_ping)
            print("Done recording data for ping #", ping_counter)

            #visualize the sonar ping
            if ping_counter >= start_view_number:
                ping_img, coor_x, coor_y = OculusPing2Img(oculus_ping, im_H, im_W, viewer_settings=viewer_settings)
                ping_timestamp = oculus_ping.get_ping_settings()['time']
                print("Creating image of ping #:", str(ping_counter), " of timestamp: ", str(ping_timestamp))
                
                if ping_img is False:
                    print("Ping image cannot be created. Something is wrong with this ping data.")
                    
                else: 
                    col_coor = np.round(coor_x[:, 0], decimals=1)
                    row_coor = np.round(coor_y[0, :], decimals=1)
                    ping_img = cv2.applyColorMap(ping_img.astype(np.uint8), opencv_colormaps[mapping])
                    
                    cv2.imshow("Sonar Ping stream", ping_img.astype(np.uint8))

                    if save_dir is not None:
                        
                        #check availability of specified output directories
                        if os.path.isdir(save_dir) == False:
                            os.makedirs(save_dir)
                        
                        ping_name = name_template + str(ping_counter) + '.jpg'
                        cv2.imwrite(save_dir + ping_name, ping_img.astype(np.uint8))
                        
                        #If distances are to be overlayed into a separate directory, 
                        if view_dist:

                            #Mark the x and y Cartesian distances represented by each pixel's x-y coordinate.
                            ping_img_range = np.copy(ping_img)
                            ping_img_range = cv2.line(ping_img_range, (0,200), (400,600), (255, 255, 255), 1)
                            ping_img_range = cv2.line(ping_img_range, (400, 600), (800, 200), (255, 255, 255), 1)
                            font = cv2.FONT_HERSHEY_SIMPLEX

                            for y in range(0, len(col_coor)):
                                if col_coor[y] % 1 == 0:
                                    ping_img_range = cv2.putText(ping_img_range, str(col_coor[y]), (5, y+2), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                                    ping_img_range = cv2.line(ping_img_range, (0, y), (5, y), (255, 255, 255) ,1)
                            for x in range(0, len(row_coor)):
                                if row_coor[x] % 1 == 0:
                                    ping_img_range = cv2.putText(ping_img_range, str(row_coor[x]), (x-6, 592), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                                    ping_img_range = cv2.line(ping_img_range, (x, 600), (x, 595), (255, 255, 255) ,1)
                            
                            #Then save into a separate distances directory from save_dir
                            save_dir_dist = save_dir[:-1] + '-w_distances/'
                            if os.path.isdir(save_dir_dist) == False: #sanity function to check for existing directory
                                os.makedirs(save_dir_dist)

                            cv2.imshow("Sonar Ping with distances", ping_img_range.astype(np.uint8))
                            cv2.imwrite(save_dir_dist + ping_name, ping_img_range.astype(np.uint8))
                        
                        print(ping_timestamp)

                        if save_timestamp:
                            timestamp_data['ping_name'].append(ping_name)
                            timestamp_data['packet_timestamp'].append(packet_timestamp)
                            timestamp_data['ping_timestamp'].append(ping_timestamp)
                        
                        if save_ping_settings:
                            dtype_id = {np.uint8:'uint8', np.int8:'int8', np.uint16:'uint16', np.int16:'int16', np.uint32:'uint32', np.int32:'int32', \
                                        np.uint64:'uint64', np.int64:'int64', np.float32:'float32', np.float64:'float64'}
                            ping_key = 'ping_siteB_' + str(ping_counter)
                            ping_dict[ping_key] = oculus_ping.ping_settings
                            ping_dict[ping_key]['dtype'] = dtype_id[ping_dict[ping_key]['dtype']]
                            ping_dict[ping_key]['watercol_pixels'] = list(oculus_ping.get_water_column_pixels().astype(object))

                    #ensure stream is alive by making next image to not wait for keypress (as normally done by using cv2.waitKey(0))
                    cv2.waitKey(2) 
            print()
            oculus_ping = None
            ping_counter += 1
            
    bin_file.close()
    cv2.destroyAllWindows()

    if save_dir is not None:

        if save_timestamp:
            timestamp_filename = 'Ping_Timestamps.csv'
            timestamp_pd = pd.DataFrame(data=timestamp_data)
            timestamp_path = save_dir + timestamp_filename
            timestamp_pd.to_csv(timestamp_path, encoding='utf-8', index=False)

        if save_ping_settings:
            json_filename = 'Ping_Settings.json'
            with open(save_dir + json_filename, 'w') as outfile:
                json.dump(ping_dict, outfile)

    print("Port number used: ", port_number)
    print()
    return collected_oculus_pings


if __name__ == "__main__":

    #In Saab's playback GUI, part of parameters that is chosen by user/viewer. Change the values here.
    viewer_data_settings = {'brightness_coeff': 4.6,
                            'gamma_correction': 3.4 ,
                            'lowlimit_Intensity': 0.0,
                            'maxlimit_Intensity': 6.0}

    #Section to view sonar pings as a contiuous stream. Use other section to view sonar ping images individually.
    #collected_pings = stream_ping_continuous(binfile_path, 600, 800, start_view_number=0, n=2, viewer_settings=viewer_data_settings, save_dir=save_dir, \
    #                                       view_dist=True, name_template ='Ping-SiteB_number_')
    
    oculus_generator = OculusPing_generator(binfile_path, n=700, start_ping=0, verbose=False)
    collected_pings = [oculus_ping for oculus_ping in oculus_generator]
    
    '''
    #test_ping = collected_pings[170] #9 #1 #58
    test_ping = collected_pings[150]
    #test_ping = collected_pings[20]
    ping_img, _, _ = OculusPing2Img(test_ping, 480, 640)
    show_image(ping_img, "Final Image", mapping='hot', blocking=False)

    test_ping = collected_pings[170]
    #test_ping = collected_pings[20]
    ping_img, _, _ = OculusPing2Img(test_ping, 480, 640)
    show_image(ping_img, "Final Image 2", mapping='hot', blocking=False)
    
    #test_ping = collected_pings[180]
    test_ping = collected_pings[180]
    ping_img, _, _ = OculusPing2Img(test_ping, 480, 640)
    show_image(ping_img, "Final Image 3", mapping='hot', blocking=True)
    #np.savetxt('./test_image.csv', ping_img, delimiter=',')
    ''' 
    
    for i in range(100, 350):#len(collected_pings)):
        test_ping = collected_pings[i]
        ping_img, _, _ = OculusPing2Img(test_ping, 480, 640)
        print("Image of ping #", i)
        #show_image(ping_img, "test", mapping='hot')
        ping_img = cv2.applyColorMap(ping_img.astype(np.uint8), opencv_colormaps['hot'])
        cv2.imshow("Sonar Ping Stream", ping_img.astype(np.uint8))
        cv2.waitKey(2)
    cv2.destroyAllWindows()
    
    #TODO: Project the current acoustic Image into (x,y,z) scatterplot coordinate, with the x, y, z values of each pixel being"
    # x = r(i) * cos(brg(i))
    # y = r(i) * sin(brg(i))
    # z = intensity(i) * ppu
    # where r(i) = maxRange * ppu. 
    # brg(i) refers to the beam bearing represented by the current pixel's column. Use brgTable this time around for reference!
    # Also, for ppu, try out both the ppu calculation used by the Saab's SAC GUI (referred to as lenofpixel in this code ), 
    # or the one proposed by Oculus in their GUI as well.


    '''
    for i in range(500, 510):
        test_ping = collected_pings[i]
        #OculusPing_viewer(test_ping, mapping='bone', verbose=True)
        test_img = test_ping.get_acousticImg_matrix()
        show_image(test_img, title="Raw Acoustic image for the current test image # {}".format(str(i)), blocking=False)
    test_ping = collected_pings[500]
    test_img = test_ping.get_acousticImg_matrix()
    show_image(test_img, "Raw acoustic image for test image # 115")
    '''


    '''
    test_ping = collected_pings[500]
    test_img = test_ping.get_acousticImg_matrix()
    test_config = test_ping.get_pingConfig()
    maxRange = test_config['M'] * test_config['range_resolution']
    test_brgTable = test_ping.get_brgTable_val()
    opening_angle_deg = test_brgTable[-1] - test_brgTable[0]
    leftover_angles = 360 - opening_angle_deg
    angleStep = opening_angle_deg / test_config['N']
    #fill out the non-filled bearings of the acoustic image to 'trick' openCV to think of a full 360 degree image
    leftover_cols = np.round(leftover_angles * (1/angleStep))
    print(opening_angle_deg, leftover_angles, leftover_cols)
    sys.exit(0)
    M = test_config['M']
    N = test_config['N']
    new_test_img = np.zeros(M, (N + int(leftover_cols)))
    test_img = test_img
    #polar_img = cv2.warpPolar(test_img, (640, 480), (320.0, 240.0), 480.0, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    polar_img = cv2.warpPolar(test_img, (640,480), (320.0, 0.0), 480.0, flags=cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG + cv2.WARP_INVERSE_MAP)
    print(polar_img, polar_img.shape)
    show_image(polar_img, "Acoustic image turned to polar coordinate", blocking=True)    
    '''
