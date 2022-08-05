
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import scipy.ndimage as ndi
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import sys
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import io,measure,color,data,filters
from xlwt import *
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import os
import glob
import pandas as pd
import dlib
import scipy.io as sio

wavelenth = 10      
orentation = 90
kernel_size = 12    
sig =5                           
gm = 0.5
ps = 0.0
th = orentation*np.pi/180
kernel = cv2.getGaborKernel((kernel_size, kernel_size), sig, th,wavelenth,gm,ps)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[2]:


def Features(mouth_centroid_x, mouth_centroid_y,b,shotname,Gabor_path):
     print(Gabor_path)
     image =cv2.imread( Gabor_path,0)
     thresh = filters.threshold_yen(image)  # high thresh
     bwimg = (image >= (thresh))  # Segmenting with threshold to generate binary image
     labels = measure.label(bwimg)  # Labeled connected region
     image_label_overlay = label2rgb(labels, image=image)
     x1=mouth_centroid_x
     y1=mouth_centroid_y
     minw = 1000
     minh = 1000
     for region in measure.regionprops(labels, intensity_image=image, coordinates='xy'):

         minr, minc, maxr, maxc = region.bbox
         if region.area >= 0:
             area = region.area
             meanintensity = region.mean_intensity
             orientation = region.orientation
             x = region.centroid[1]
             y = region.centroid[0]
             w = abs(x1 - x)
             h = abs(y - y1)

         if w < minw and h < minh:
             minw = w
             minh = h
             min_maxc = maxc
             min_maxr = maxr
             min_minc = minc
             min_minr = minr
             min_area = region.area
             min_meanintensity = meanintensity
             min_orientation = orientation
             min_centroidx = x
             min_centroidy = y

     # write the weight, height, area ,mass to txt
     width = min_maxc - min_minc
     height = min_maxr - min_minr
     Final_area = min_area
     Final_meanintensity = min_meanintensity
     Final_orientation = min_orientation
     Final_centroidx = min_centroidx 
     Final_centroidy = min_centroidy

     gabor_data = np.zeros((7))
     gabor_data[0] = width
     gabor_data[1] = height
     gabor_data[2]= Final_area 
     gabor_data[3]=Final_centroidx 
     gabor_data[4]=Final_centroidy
     gabor_data[5]=Final_meanintensity 
     gabor_data[6]=Final_orientation 
     #print('-------------')
     #print(gabor_data)
     return gabor_data


# In[3]:


# Define input and output path
speaker = 's9'


inputPath ='D:/Mrs_backup/speech_test/video/'+speaker+'/'

GaborPicturePath = 'D:/Mrs_backup/speech_test/gabor_picture/'+speaker+'/'
if not os.path.exists(GaborPicturePath):
    os.mkdir(GaborPicturePath)
    
GaborFeaturePath = 'D:/Mrs_backup/speech_test/gabor_feature/'+speaker+'/'
if not os.path.exists(GaborFeaturePath):
    os.mkdir(GaborFeaturePath)

j = 0
for video in glob.glob(inputPath+'*.mp4'): 
    (filepath, tempfilename) = os.path.split(video)
    (shotname, extension) = os.path.splitext(tempfilename)
        
    gabor_path = os.path.join(GaborPicturePath, shotname)
    if not os.path.exists(gabor_path):
        os.mkdir(gabor_path)
            
    cap = cv2.VideoCapture(video)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_length= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gabor_input = np.zeros((video_length,7))
    i=0
    while (cap.isOpened()):
        i=i+1
        ret, frame = cap.read()
        if ret == True:
                
            dets = detector(frame, 1)
                
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                mouth_centroid_x = shape.part(48).x + abs(shape.part(54).x - shape.part(48).x) / 2 - shape.part(5).x
                mouth_centroid_y = shape.part(48).y + abs(shape.part(54).y - shape.part(48).y) / 2 - shape.part(13).y
                ROI_mouth = frame[shape.part(13).y:shape.part(10).y, shape.part(5).x:shape.part(11).x]

                #imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                imgGray = cv2.cvtColor(ROI_mouth, cv2.COLOR_BGR2GRAY)  
                imgGray_f = np.array(imgGray,dtype=np.float32)   
                imgGray_f /=255.

                dest = cv2.filter2D(imgGray_f, cv2.CV_32F, kernel)
                format_i="{number:02}".format(number=i)
                Gabor_Path = gabor_path + '/'+format_i+ '.jpg'
                cv2.imwrite(Gabor_Path, np.power(dest, 2))

                gabor_data=Features(mouth_centroid_x, mouth_centroid_y,i,shotname,Gabor_Path)
                gabor_input[i-1,:]= gabor_data
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

    gabor_data= GaborFeaturePath+shotname+'.mat'
    sio.savemat(gabor_data, mdict={'gabor_input':gabor_input})
    j=j+1
    print(j, end=' ')

