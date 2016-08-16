import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import math


def create_database(directory, show = True):
    '''
    Process all images in the given directory.
    Every image is cropped to the detected face, resized to 100x100 and save in another directory (orignal directory name + "2").
    
    @param directory:    directory to process
    @param show:         bool, show all intermediate results
    
    '''
    #load a pre-trained classifier
    cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    #loop through all files
    for filename in fnmatch.filter(os.listdir(directory),'*.jpg'):
        file_in = directory+"/"+filename
        file_out= directory+"2/"+filename
        img = cv2.imread(file_in)
        if show:
            cv2.imshow('img',img)
            cv2.waitKey(0)
        #do face detection    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
        rects[:,2:] += rects[:,:2]
        rects = rects[0,:]
        #show detected result
        if show:
            vis = img.copy()
            cv2.rectangle(vis, (rects[0], rects[1]), (rects[2], rects[3]), (0, 255, 0), 2)
            cv2.imshow('img', vis)
            cv2.waitKey(0)

        #crop image to the rectangle and resample it to 100x100 pixels
        result = img[rects[1]:rects[3], rects[0]:rects[2]]                  #y1:y2, x1:x2
        result = cv2.resize(result, (100, 100), interpolation=cv2.INTER_CUBIC)
        
        #show result
        if show:
            cv2.imshow('img', result)
            cv2.waitKey(0)            
        #save the face in a second directory
        cv2.imwrite(file_out, result)
    cv2.destroyAllWindows()



def createX(directory,nbDim=10000):
    '''
    Create an array that contains all the images in directory.
    @return np.array, shape=(nb images in directory, nb pixels in image)
    '''
    filenames = fnmatch.filter(os.listdir(directory),'*.jpg')
    nbImages = len(filenames)
    X = np.zeros( (nbImages,nbDim) )#, dtype=np.uint8 )
    for i,filename in enumerate( filenames ):
        file_in = directory+"/"+filename
        img = cv2.imread(file_in)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        X[i,:] = gray.flatten()
    print X.dtype
    return X


def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    X = X - mu
    return W.transpose().dot(X)

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    return mu + W.dot(Y)

def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n

    # compute mean
    mean = X.mean(0)

    # normalize input matrix
    norm_X = X - mean
    #C = norm_X.dot(norm_X.transpose())     # covariance matrix
    #print C.shape

    #compute covariance matrix n x n
    cov_matrix = np.cov(X)

    #compute eignevalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # TEST eigenvectors
    for i in range(len(eigenvalues)):
        eigv = eigenvectors[:, i].reshape(1, n).T   # columns
        np.testing.assert_array_almost_equal(cov_matrix.dot(eigv), eigenvalues[i] * eigv, decimal=6, err_msg="Not equal")

    #create eigenvalues and eigenvectors pairs and sort them
    eig_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eig_pairs.sort()
    eig_pairs.reverse()

    # choose only k leargest eigenvalues with eigenvectors
    k_eigenvalues = []
    k_eigenvectors = []
    for i in range(nb_components):
        #print(eig_pairs[i][0])
        #print eig_pairs[i][1]
        k_eigenvectors = np.hstack((k_eigenvectors, eig_pairs[i][1]))
        k_eigenvalues = np.hstack((k_eigenvalues, eig_pairs[i][0]))

    k_eigenvectors = k_eigenvectors.reshape(nb_components, k_eigenvectors.shape[0]/nb_components)
    #print len(eigenvalues), nb_components, k_eigenvectors.shape[0]
    #print k_eigenvectors.shape, k_eigenvectors
    #print k_eigenvalues.shape, k_eigenvalues

    # reconstruct input
    print k_eigenvectors.shape, X.shape
    k_eigenvectors = k_eigenvectors.dot(X)
    print k_eigenvectors.shape
    k_eigenvectors = normalize_vector(k_eigenvectors)
    print k_eigenvectors.shape
    return k_eigenvalues, k_eigenvectors.transpose(), mean



def normalize_vector(vector):
    '''
    Normalize vector.
    @param vector:
    @return:
    '''
    eigenvectors = vector
    res = []
    for i in range(eigenvectors.shape[0]):
        magnitude = math.sqrt(sum([x**2 for x in eigenvectors[i]]))
        res.append([x/magnitude for x in eigenvectors[i]])
    return np.array(res)


def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)

if __name__ == '__main__':
    #create database of normalized images
    for directory in ["data/arnold", "data/barack"]:
        create_database(directory, show = False)
    
    show = True
    
    #create big X arrays for arnold and barack
    Xa = createX("data/arnold2")
    Xb = createX("data/barack2")
            
    #Take one part of the images for the training set, the rest for testing
    nbTrain = 6
    Xtest = np.vstack( (Xa[nbTrain:,:],Xb[nbTrain:,:]) )
    Ctest = ["arnold"]*(Xa.shape[0]-nbTrain) + ["barack"]*(Xb.shape[0]-nbTrain)
    Xa = Xa[0:nbTrain,:]
    Xb = Xb[0:nbTrain,:]

    #do pca
    [eigenvaluesa, eigenvectorsa, mua] = pca(Xa,nb_components=6)
    [eigenvaluesb, eigenvectorsb, mub] = pca(Xb,nb_components=6)
    #visualize
    cv2.imshow('img',np.hstack( (mua.reshape(100,100),
                                 normalize(eigenvectorsa[:,0].reshape(100,100)),
                                 normalize(eigenvectorsa[:,1].reshape(100,100)),
                                 normalize(eigenvectorsa[:,2].reshape(100,100)))
                               ).astype(np.uint8))
    cv2.waitKey(0) 
    cv2.imshow('img',np.hstack( (mub.reshape(100,100),
                                 normalize(eigenvectorsb[:,0].reshape(100,100)),
                                 normalize(eigenvectorsb[:,1].reshape(100,100)),
                                 normalize(eigenvectorsb[:,2].reshape(100,100)))
                               ).astype(np.uint8))
    cv2.waitKey(0) 
            
    nbCorrect = 0
    for i in range(Xtest.shape[0]):
        X = Xtest[i,:]
        
        #project image i on the subspace of arnold and barack
        Ya = project(eigenvectorsa, X, mua )        #Szelinski - result of this is a, u is eigenvectors matrix
        Xa= reconstruct(eigenvectorsa, Ya, mua)
        
        Yb = project(eigenvectorsb, X, mub )
        Xb= reconstruct(eigenvectorsb, Yb, mub)
        if show:
            #show reconstructed images
            cv2.imshow('img',np.hstack( (X.reshape(100,100),
                                         np.clip(Xa.reshape(100,100), 0, 255),
                                         np.clip(Xb.reshape(100,100), 0, 255)) ).astype(np.uint8) )
            cv2.waitKey(0)   

        #classify i
        if np.linalg.norm(Xa-Xtest[i,:]) < np.linalg.norm(Xb-Xtest[i,:]):
            bestC = "arnold"
        else:
            bestC = "barack"
        print(str(i)+":"+str(bestC))
        if bestC == Ctest[i]:
            nbCorrect+=1
    
    #Print final result
    print(str(nbCorrect)+"/"+str(len(Ctest)))
    
