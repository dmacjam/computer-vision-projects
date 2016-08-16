'''
Gaussian smoothing with Python.
'''

import cv2
import numpy as np
import math
import os

def gaussian_filter(sigma, filter_length=None):
    '''
    Given a sigma, return a 1-D Gaussian filter.
    @param     sigma:         float, defining the width of the filter
    @param     filter_length: optional, the length of the filter, has to be odd
    @return    A 1-D numpy array of odd length, 
               containing the symmetric, discrete approximation of a Gaussian with sigma
               Summation of the array-values must be equal to one.
    '''
    if filter_length==None:
        #determine the length of the filter
        filter_length= math.ceil(sigma*5) 
        #make the length odd
        filter_length= 2*(int(filter_length)/2) +1   
    
    #make sure sigma is a float
    sigma=float(sigma)
    
    #create the filter
    result = np.zeros( filter_length )

    mean = int(filter_length/2)
    for i in range(filter_length):
        result[i] = (1 / (math.sqrt(2*math.pi)*sigma)) * math.exp( -(i-mean)**2/(2.0*sigma**2) )

    print result

    # normalise filter (sum = 1, else you lose intensity values)
    result = result/np.sum(result)

    print result

    #return the filter
    return result


def test_gaussian_filter():
    '''
    Test the Gaussian filter on a known input.
    '''
    sigma = math.sqrt(1.0/2/math.log(2))
    f = gaussian_filter(sigma, filter_length=3)
    correct_f = np.array([0.25, 0.5, 0.25])
    error = np.abs( f - correct_f)
    if np.sum(error)<0.001:
        print "Congratulations, the filter works!"
    else:
        print "Still some work to do.."
        
      

    
def gaussian_smooth1(img, sigma): 
    '''
    Do gaussian smoothing with sigma.
    Returns the smoothed image.
    '''
    result = np.zeros_like(img)
    
    #get the filter
    filter = gaussian_filter(sigma)
    
    #smooth every color-channel
    for c in range(3):
        #smooth the 2D image img[:,:,c]
        #tip: make use of numpy.convolve
        # convolve: Returns the discrete, linear convolution of two one-dimensional sequences
        
        # process each row of the image
        for i in range(img.shape[0]):
            result[i,:,c] = np.convolve(img[i,:,c], filter, mode='same') #or mode='valid'
            
        # process each column of the image
        for j in range(img.shape[1]):
            result[:,j,c] = np.convolve(result[:,j,c], filter, mode='same') #or mode='valid'
        
        
    
    return result

#this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    #clear screen
    #os.system('clear');
    
    #test the gaussian filter
    test_gaussian_filter()

    #read an image
    img = cv2.imread('face.tiff')

    #print the dimension of the image
    print img.shape
    
    #show the image, and wait for a key to be pressed
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #smooth the image
    smoothed_img = gaussian_smooth1(img, 2)
    
    #show the smoothed image, and wait for a key to be pressed
    cv2.imshow('smoothed_img',smoothed_img)
    cv2.waitKey(0)
    
