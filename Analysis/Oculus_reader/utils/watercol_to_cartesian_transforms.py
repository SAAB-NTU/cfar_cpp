import sys
import os

from matplotlib import pyplot
import numpy as np
import skimage
from skimage.util import img_as_float
from skimage.transform import rotate
from scipy.ndimage import map_coordinates

from scipy.interpolate import interp1d
import cv2
import copy

from matplotlib import pyplot as plt

def pyplot_fig(img, cmap="hot", title=None, figsize=(11,5)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    
    if title is not None:
        plt.title(title)
    
    plt.show()

def scipy_skimage_watercol_to_cartesian(raw_img, sonar_fov, out_H=None, out_W=None, spline_order=1):
    """
    #Remapping function based on inverse warp_polar implementation found in:
    #https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/2
    """
    #determine cartesian dimensions of new image
    r = raw_img.shape[0]
    new_img = np.zeros((r * 2, r * 2), dtype=raw_img.dtype)
    cart_center = np.array(new_img.shape) / 2 - 0.5 #0.5 to eliminate un-even numbers
    out_h, out_w = new_img.shape

    sonar_fov_rad = np.radians(sonar_fov)
    
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

    #ts[ys<0] = - (sonar_fov_rad - ts[ys<0]) #np.pi*2 - ts[ys<0] #adjustment mapping for negative theta. Not necessary in our case 
    
    #scale the theta value against raw_img's opening angle value (in rad) 
    ts *= (raw_img.shape[1]-1)/sonar_fov_rad  

    #Remap using scipy's map_coordinate function
    map_coordinates(raw_img, (rs, ts), output=new_img, order=spline_order)

    #Rotation & slicing needed as map_coordinate's 0 theta starts at 3 o'clock position.
    new_img = rotate(new_img, -(90 + (sonar_fov/2) ))
    new_img = new_img[r:, r - int(r * np.sin(sonar_fov_rad / 2)):r + int(r * np.sin(sonar_fov_rad/2))]

    #Sanity check to see whether user requested a specific Height/Width value for new image. 
    # If there aren't any requests, then just return as is
    if (out_H is not None) and (out_W is not None):
        new_img = skimage.transform.resize(new_img, (out_H, out_W))

    polar2cart_x, polar2cart_y = rs, ts


    #NOTE: the images provided by the scipy and skimage transformation by default are in float64 values (max val = 1.0)
    #Hence the multiplication w/ 255 before turning to integer value. May not suit application depending on what you want. 
    return (new_img * 255.0).astype(np.uint8), polar2cart_x, polar2cart_y


def omsf_watercol_to_cartesian(raw_img, sonar_range, range_res=None, bearings_deg=None):
    """
    Transformation using SciPy and OpenCV based on method written by the original OMSF author. Seem to be more reliable overall.
    """

    def deg_to_rad(degrees):
        return degrees / 180.0 * np.pi

    #Helper blocks of code in the Original implementation of the OMSF method by JMcConnell, modified to suit the test environment format used in NTU UWR project
    def generate_map_xy(hori_img, sonar_range, range_res=None, bearings_deg=None):
        # type: (OculusPing) -> None
        """Generate a mesh grid map for the sonar image.
        Keyword Parameters:
        ping -- a OculusPing message
        """
        REVERSE_Z = 1
        rows, bearings = hori_img.shape
        new_rows = rows
        
        
        #NOTE: Actual oculus sonar has their own bearings and range resolution values. For simulator images can use
        #self-calculated bearing and range resolution values as provided by the simulator image data 
        if bearings_deg is None:
            bearings_deg = np.asarray(range(0, bearings)) - bearings/2.0
        
        bearings_rad = deg_to_rad(bearings_deg)

        if range_res is None:
            range_res = sonar_range / rows
        
        #Overall mathematics similar to the inverse warp_polar function that we referenced for 
        #visualize_oculus_pings.py, https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/2.
        #Note especially the similarity of how we obtain the new image's width, while preserving the original
        #raw /watercolumn image's height
        new_height = new_rows * range_res
        new_width = np.sin(deg_to_rad(bearings_deg[-1] - bearings_deg[0]) / 2.0) * new_height * 2
        new_cols = int(np.ceil(new_width / range_res))
        
        #approximate 1d interpolated function for estimating the image column from inputs of bearing values, 
        #based on the pixel column vs. bearing values (in radians) that we have
        f_bearings = interp1d(bearings_rad, range(len(bearings_rad)), kind="linear", \
                                bounds_error=False, fill_value=-1, assume_sorted=True)
        
        xx,yy = np.meshgrid(range(new_cols), range(new_rows))
        x = range_res * (new_rows - yy)
        y = range_res * ((-new_cols / 2.0) + xx + 0.5)
        b = np.arctan2(y, x) * REVERSE_Z
        r = np.sqrt(np.square(x) + np.square(y))
        map_y = np.asarray(r / range_res, dtype=np.float32)
        map_x = np.asarray(f_bearings(b), dtype=np.float32)

        
        return map_x, map_y, new_width, new_height #new_cols, new_rows

    map_x, map_y, cart_width, cart_height = generate_map_xy(raw_img, sonar_range, range_res=range_res, bearings_deg=bearings_deg)

    new_img = cv2.remap(raw_img, map_x, map_y, cv2.INTER_LINEAR)

    return new_img

