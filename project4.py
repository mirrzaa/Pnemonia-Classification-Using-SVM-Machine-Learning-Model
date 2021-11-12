# -*- coding: utf-8 -*-
"""

# Multimedia Computing Project
# Digital Image Processing/Computer Vision/Machine Learning
# 
# Done by: U1610146 Mirzashomol Karshiev
#
# Image Classification using SVM machine learning algorithm
#
# Classifies the images of COVID-19 whether person has virus or not
#
# negative folder contains -->pictures of normal lungs
# positive folder contains -->pictures of pneumonia  


"""
import cv2
import numpy as np
import os
import tensorflow as tf
from skimage.feature import hog
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

DATADIR="C:/Users/HP/Desktop/MC Project/images"
categories=["negative","positive"]

class ImageProcessing:
    
    def __init__(self):
        pass
    
    #<-------------- Imge Denoising----------------->
    def __imageProcessing__(self,image):
        
        #image denoising using GausianBlur
        blur_image=cv2.GaussianBlur(image,(5,5),0)
        
        #histogram oriented gradient for feature extracting can be used
        
        return blur_image
    
    #<--------------Sobel Filtering----------------->
    def __edgeDetection__(self,img):
        # Calcution of Sobelx
        sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5) 
        #absolute of sobelx gradient
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        
        # Calculation of Sobely 
        sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)
        
        #absolute of sobely gradient
        abs_sobely = cv2.convertScaleAbs(sobely)
        
        #joining together sobelx and sobely
        cv2.addWeighted(abs_sobelx,0.5,abs_sobely,0.5,0,img)
        
        return img
    
    #<--------------Vectorizing Image----------------->
    def __resizeTo1xN__(self,img):
        try:
            #resize the image
            img=cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
            
            #get the image width and height
            w,h=img.shape
            
            return img.reshape(1,w*h)   
        except Exception:
            pass
            #get the image width and height
        
        
    #<--------------Loading Images for Training----------------->
    def __populateTrainMatrix__(self):
        
        training_data=[]
        
        for category in categories:
            path= os.path.join(DATADIR,category)
            for img in os.listdir(path):
                #reading image from folder
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                #denoising image
                denoised_img=self.__imageProcessing__(img_array)
                #edge detection using Sobel
                img_edge=self.__edgeDetection__(denoised_img)
                #resizing to 1xN vector
                resized_img=self.__resizeTo1xN__(img_edge)
                #collecting all vectorized images for training
                training_data.append(resized_img)
        
        return training_data
    
    #<--------------Labeling Train Images----------------->
    def __populateLabels__(self):
        list=[]
        
        #2 folders negative positive
        for i in range(2):          
            for j in range(10):
                #labeling wiht 0 and 1 for classification
                list.append(i)
                
        #converting list in numpy array
        labels=np.array(list)
        
        return labels
    
    #<--------------SVM Model Initialization----------------->
    def __modelSVM__(self):
        #create SVM machine learning classifier
        svm = cv2.ml.SVM_create()
        #SVC
        svm.setType(cv2.ml.SVM_C_SVC)
        #kernel type Linear means linear regression
        svm.setKernel(cv2.ml.SVM_LINEAR)
        #criteria
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        
        return svm
    
    #<--------------Training SVC Model----------------->    
    def __modelFitData__(self,x_train,y):
        #calling model
        model=self.__modelSVM__()
        #fitting training set
        model.train(x_train, cv2.ml.ROW_SAMPLE, y)
        
        return model
        
    #<--------------Testing On Sample Image----------------->    
    def __testing__(self):
        #test image reading
        test_image = cv2.imread('images/test_images/person6_bacteria_22.jpeg')
        #converting image to gray scale
        test_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
        
        #detection edges of test image
        image=self.__edgeDetection__(test_image)
        #vectorizing test image 1xN and reshaping for predict model
        vector_image=self.__resizeTo1xN__(image).reshape(1,65536)
        #converting from int to float32
        vector_image=np.float32(vector_image)
        
        #trainnig matrix is received
        training_matrix=self.__populateTrainMatrix__()
        #reshaping training matrix for model fitting
        train=np.array(training_matrix).reshape(20,65536)
        train=np.float32(train)
        #lables are received
        labels=self.__populateLabels__()
        #fitting training data
        modelSVC=self.__modelFitData__(train, labels)
       
        # Data for visual representation
        width = 512
        height = 512
        #create dark image wiht size 512x512
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        red = (0,0,255)
        blue = (255,0,0)
        
        bool=0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                #predict test image
                response = modelSVC.predict(vector_image)[1] 
                
                if response == 1: # if image is COVID-19 image will be red
                    image[i,j] = red
                    bool=1
                elif response == 0:# if image is not COVID-19 image will be blue
                    image[i,j] = blue
                    bool=0

        #putting text on image
        if bool==1:
            cv2.putText(image,'POSITIVE CASE FOR COVID-19',(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        else:
            cv2.putText(image,'NEGATIVE CASE FOR COVID-19',(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
        
        cv2.imshow('Result of test',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
#main function
if __name__ == '__main__':
    ob = ImageProcessing()
    ob.__testing__()
 

    #img = cv2.imread('images/negative/1.jpeg')
    #res_img=cv2.resize(img,(512,512))
    #blur_image=cv2.GaussianBlur(res_img,(5,5),0)
    #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #fd, hog_image = hog(res_img, orientations=9, pixels_per_cell=(8, 8), 
    #cells_per_block=(2, 2), visualize=True, multichannel=True)
    

    #cv2.imshow('Denoising',blur_image)
    #cv2.imshow('hog_image',hog_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
