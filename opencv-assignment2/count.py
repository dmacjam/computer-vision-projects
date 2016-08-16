'''
Cell counting.

'''

import cv2
import cv2.cv
import numpy as np
import matplotlib.pyplot as plt
import math

def detect(img):
    '''
    Do the detection.
    '''
    #create a gray scale version of the image, with as type an unsigned 8bit integer
    img_g = np.zeros( (img.shape[0], img.shape[1]), dtype=np.uint8 )
    img_g[:,:] = img[:,:,0]

    #1. Do canny (determine the right parameters) on the gray scale image
    edges = cv2.Canny(img, 90, 140) #TODO!
    
    #Show the results of canny
    canny_result = np.copy(img_g)
    canny_result[edges.astype(np.bool)]=0
    cv2.imshow('img',canny_result)
    cv2.waitKey(0)

    #2. Do hough transform on the gray scale image - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html#hough-circles
    circles = cv2.HoughCircles(img_g, cv2.cv.CV_HOUGH_GRADIENT, dp=1, minDist=20, param1=105, param2=21, minRadius=20, maxRadius=80  ) #TODO!
    circles = circles[0,:,:]
    
    #Show hough transform result
    showCircles(img, circles)

    #return circles

    #3.a Get a feature vector (the average color) for each circle
    nbCircles = circles.shape[0]
    features = np.zeros( (nbCircles,3), dtype=np.int)
    for i in range(nbCircles):
        features[i,:] = getAverageColorInCircle( img , int(circles[i,0]), int(circles[i,1]), int(circles[i,2]) ) #TODO!
    
    #3.b Show the image with the features (just to provide some help with selecting the parameters)
    showCircles(img, circles, [ str(features[i,:]) for i in range(nbCircles)] )

    #return circles

    #3.c Remove circles based on the features
    selectedCircles = np.zeros( (nbCircles), np.bool)
    for i in range(nbCircles):
        if features[i, 0]> 170 and features[i,0] < 203\
                and features[i,1]> 170 and features[i,1 < 215\
                and features[i,2]> 170 and features[i,2] < 215]:    #TODO
            selectedCircles[i]=1
    circles = circles[selectedCircles]

    #Show final result
    showCircles(img, circles)    
    return circles
        
    
def getAverageColorInCircle(img, cx, cy, radius):
    '''
    Get the average color of img inside the circle located at (cx,cy) with radius.
    '''
    maxy,maxx,channels = img.shape      #return rows, columns
    nbVoxels = 0
    C = np.zeros( (3) )

    #TODO!
    mask = np.zeros( (img.shape[0], img.shape[1]), dtype=np.uint8 )

    # Y - row, X - column
    for i in range(cy-radius, cy+radius):
        for j in range(cx-radius, cx+radius):
            if(i >= 0 and i < maxy and j >= 0 and j < maxx):
                mask[i][j] = 1

    A = cv2.mean(img, mask)     # calculates averages for every channel in mask
    C = A[:-1]

    return C
    
    
    
def showCircles(img, circles, text=None):
    '''
    Show circles on an image.
    @param img:     numpy array
    @param circles: numpy array 
                    shape = (nb_circles, 3)
                    contains for each circle: center_x, center_y, radius
    @param text:    optional parameter, list of strings to be plotted in the circles
    '''
    #make a copy of img
    img = np.copy(img)
    #draw the circles
    nbCircles = circles.shape[0]
    for i in range(nbCircles):
        cv2.circle(img, (int(circles[i,0]), int(circles[i,1])), int(circles[i,2]), cv2.cv.CV_RGB(255, 0, 0), 2, 8, 0 )
    #draw text
    if text!=None:
        for i in range(nbCircles):
            cv2.putText(img, text[i], (int(circles[i,0]), int(circles[i,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2.cv.CV_RGB(0, 0,255) )
    #show the result
    cv2.imshow('img',img)
    cv2.waitKey(0)    


        
if __name__ == '__main__':
    #read an image
    img = cv2.imread('normal.jpg')
    
    #print the dimension of the image
    print img.shape
    
    #show the image
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    #do detection
    circles = detect(img)
    
    #print result
    print "We counted "+str(circles.shape[0])+ " cells."
    








