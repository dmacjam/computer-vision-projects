'''
Gaussian smoothing with Python and OpenCV.
'''

import cv2
import math
import numpy as np
   
def gaussian_smooth2(img, sigma): 
    '''
    Do gaussian smoothing with sigma.
    Returns the smoothed image.
    '''
    result = np.zeros_like(img)
    
    #determine the length of the filter
    filter_length= math.ceil(sigma*5) 
    #make the length odd
    filter_length= 2*(int(filter_length)/2) +1  
            
    #Go ahead! 
    #Tip: smoothing=blurring, a filter=a kernel
    result = cv2.GaussianBlur(img, (filter_length,filter_length), sigma)
    
    return result



#this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    #read an image
    img = cv2.imread('face.tiff')
    
    #show the image, and wait for a key to be pressed
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #smooth the image
    smoothed_img = gaussian_smooth2(img, 2)
    
    #show the smoothed image, and wait for a key to be pressed
    cv2.imshow('smoothed_img',smoothed_img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    