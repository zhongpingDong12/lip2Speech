
# coding: utf-8

# (1) Import libararies

# In[1]:


import cv2
import glob
import dlib
import numpy as np
import scipy.io as sio
from keras import backend as BK
from keras.models import load_model
import os
import fnmatch
import numpy as np
from scipy.ndimage.interpolation import map_coordinates


# In[2]:


def getInputSize(inputPath):
    firstFilePath = inputPath + os.listdir(inputPath)[0]
    dataSize=sio.loadmat(firstFilePath)
    return dataSize

def interp_Frame(A):
    #print("A.shape IS ",A.shape)
    
    new_dims = []
    for original_length, new_length in zip(A.shape, (num_features*2+2,visualLength+2)):
        new_dims.append(np.linspace(0, original_length-1, new_length))

    coords = np.meshgrid(*new_dims, indexing='ij')
    B = map_coordinates(A, coords)
    #print('B.shape IS ',B.shape)
    return B


# Define diff function to add 3 time-derivative channels
def diff(buf_input):
    buf_input=np.pad(buf_input,((0,0),(1,0)),'edge')
    buf_output=np.diff(buf_input,axis=1)
    #print(buf_output.shape)
    return buf_output

# Define slice_video_3D function to divide into 15 non-overlap slices each of length 5
def slice_video_3D(video):
    video_output =np.empty((num_slices,2,(num_features*2+2),slice_length), np.dtype('float32'))
    
    start=0
    for i in range(0,num_slices):
        video_output[i,:,:,:]=video[:,:,start:start+slice_length]
        start+=slice_length
    return video_output

def getVisualInput(video_input,visualPath):
    start=0
    for count, filename in enumerate(sorted(os.listdir(visualPath)),start=0): 
        visualFile  =visualPath+ filename 
        mat=sio.loadmat(visualFile)['gabor_input']
        interp_mat = interp_Frame(mat.T)
        
        diff_video=np.empty((2,interp_mat.shape[0],interp_mat.shape[1]))
        diff_video[0,:,:]=interp_mat
        diff_video[1,:,:]=diff(interp_mat)
        
        # Call slice_video_3D function
        data_vid=slice_video_3D(diff_video)
        
        # Add total number of slices 
        video_input[start:start+num_slices,:,:,:]=data_vid
        start+=num_slices
    
    video_input = np.reshape(video_input,(video_input.shape[0],video_input.shape[3],video_input.shape[1]*video_input.shape[2]))
    return video_input


# In[3]:


# Define input and output path
visualInputPath ='D:/Mrs_backup/speech_test/all_vocabulary/06_GaborFeature/'
outputPath = 'D:/Mrs_backup/speech_test/all_vocabulary/07_GaborFeatureCollection/'
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
    
word_list = os.listdir(visualInputPath)
visualInputSize = getInputSize(visualInputPath+word_list[0]+'/')
visualLength = visualInputSize['gabor_input'].shape[0]
visualFeature = visualInputSize['gabor_input'].shape[1]
print('Visual input length is:' + str(visualLength))
print('Visual feature is:' + str(visualFeature))

num_slices=2
num_features=7
slice_length = int((visualLength+2)/num_slices) 

for i in range (0,len(word_list)):
    visualPath = visualInputPath+word_list[i]+'/'
    dataOutputPath=outputPath+word_list[i]+'/'
    if not os.path.exists(dataOutputPath):
        os.mkdir(dataOutputPath)
    
    # Load file and get the size
    numfiles=len(fnmatch.filter(os.listdir(visualPath), '*.mat'))
    print('Total number of word {'+ word_list[i] +'} is: ' + str(numfiles))
    
    #Collect visual input
    visual_input =np.empty((numfiles*num_slices,2,int(2*num_features+2),slice_length), np.dtype('float32'))
    visual_input = getVisualInput(visual_input,visualPath)
    print("Visual input shape is: "+ str(visual_input.shape))
    sio.savemat(dataOutputPath+'visualInput.mat', mdict={'visual_input': visual_input})

