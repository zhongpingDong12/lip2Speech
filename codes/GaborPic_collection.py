
# coding: utf-8

# (1) Import libraries

# In[1]:


import os
import fnmatch
import cv2
import numpy as np
import sys
np.random.seed(1337)  # for reproducibility
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential, Model, load_model
from keras import backend as BK
import cv2
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.io as sio
from collections import OrderedDict
import random 
random.seed(100)
# %pylab inline
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# (2) Define variables and parameters

# In[2]:


word = 'z'

num_videos = 38

num_test=10


num_train=num_videos-num_test
num_img =126
img_width,img_height,num_slices =28,28,2
slice_length = int(num_img/num_slices)
print(slice_length)


base_path='D:/Mrs_backup/speech_test/all_vocabulary/gp_model/gp/'+ word + '/'
audio_path ='D:/Mrs_backup/speech_test/all_vocabulary/gp_model/audio_spec/'+ word + '/'
autoencoder_path= 'D:/Mrs_backup/speech_test/all_vocabulary/04_autoencoder/'+ word + '/'
specTxtfile='D:/Mrs_backup/speech_test/all_vocabulary/gp_model/spec_txt/'+ word+'_valid_aud_specs.txt'

integrate_data_path ='D:/Mrs_backup/speech_test/all_vocabulary/gp_model/modelInput/'+ word + '/'
if not os.path.exists(integrate_data_path):
    os.mkdir(integrate_data_path)


# (3) Define functions for video input

# In[3]:


# Define diff function to add 3 time-derivative channels
def diff(buf_input):
    buf_input=np.pad(buf_input,((0,0),(0,0),(1,0)),'edge')
    buf_output=np.diff(buf_input,axis=2)
    #print(buf_output.shape)
    return buf_output

# Define slice_video_3D function to divide into 15 non-overlap slices each of length 5
def slice_video_3D(video):
    video_output =np.empty((num_slices,3,img_height,img_width,slice_length), np.dtype('float32'))
    
    start=0
    for i in range(0,num_slices):
        video_output[i,:,:,:,:]=video[:,:,:,start:start+slice_length]
        start+=slice_length
    return video_output

# Define diff function to add 3 time-derivative channels
def diff(buf_input):
    buf_input=np.pad(buf_input,((0,0),(0,0),(1,0)),'edge')
    buf_output=np.diff(buf_input,axis=2)
    #print(buf_output.shape)
    return buf_output


# In[4]:


# Define the shape of the video_input [10*15,40,80,5]
video_input =np.empty((num_videos*(num_slices),3,int(img_height),int(img_width),int(slice_length)), np.dtype('float32'))
print(video_input.shape)


# In[5]:


# #D:\speech_dataset\GRID\GP_100_75frames\s1_gp_test\gabor\bbal8p
# img_path = base_path +'002/27.jpg'
# img = mpimg.imread(img_path)
# imgplot = plt.imshow(img)
# plt.show()

# image = cv2.resize(img,(80,40))
# print(image)
# plt.imshow(image)


# In[6]:


start = 0
print(num_videos)
for n in range(0,num_videos):
    print(n, end=' ')
    format_num1="{number:03}".format(number=n)
    path = base_path+str(format_num1)
    dirs = os.listdir(path)
    num_img = len(dirs) 
    #print(num_img)
    
    tmp0 = np.empty((int(img_height),int(img_width),int(num_img)), np.dtype('float32'))
   
    for i in range(0,num_img):
        img_path =  path + '/'+str(dirs[i])
        #print(img_path)
        
        
        # Load an color image in grayscale
        img = cv2.imread(img_path,0)
        
#         image = cv2.resize(img,(80,40))
        image = cv2.resize(img,(28,28))
        #print(image.shape)
        
        tmp0[:,:,i] = image
        #print(tmp0)
    
    # Call diff function to add 3 time-derivative channels
    diff_video=np.empty((3,tmp0.shape[0],tmp0.shape[1],tmp0.shape[2]))
    diff_video[0,:,:,:]=tmp0
    diff_video[1,:,:,:]=diff(tmp0)
    diff_video[2,:,:,:]=diff(diff_video[1,:,:,:])
    #print(diff_video.shape)
    
    # Call slice_video_3D function
    data_vid=slice_video_3D(diff_video)
    #data_vid=data_vid/255
    #print(data_vid)
    
    # Add total number of slices 
    video_input[start:start+num_slices,:,:,:,:]=data_vid
    #print(video_input.shape)
    start+=num_slices
    #print(start)
    
print('Video slices shape:'+str(video_input.shape))


# In[7]:


# print(video_input.shape)
# print(video_input[100,0,:,:,1])


# (5) Define functions for audio data

# In[8]:


# Define slice_audio_spec function to to divide into 15 non-overlap slices each of length 26
def slice_audio_spec(audio_spec):
    global AUDIO_LENGTH
    window_size=int(AUDIO_LENGTH/num_slices) #from time to number of audio index 
    #print(window_size)
    audio_output =np.empty((num_slices,audio_spec.shape[0],window_size), np.dtype('float32'))
    start=0
    for i in range(0,num_slices):
        audio_output[i,:,:]=audio_spec[:,start:start+window_size]
        start+=window_size
        if start>AUDIO_LENGTH-window_size:
            break
    #print(audio_output.shape)
    #print(audio_output[1,:,1])
    return audio_output

# Define get activations function # Extract the 32-bin bottleneck features as target for the main network
def get_activations(model, layer_in, layer_out, X_batch):
    get_activations = BK.function([model.layers[layer_in].input, BK.learning_phase()], [model.layers[layer_out].output])
    activations = get_activations([X_batch,0])
    return activations

#Define padding function
def get_padded_spec(data):
    
    # Compress the spectrogram by raising to the power 1/3
    #print(data[1:3,1])
    data=np.power(data,.3)
    #print(data[1,370])
    
    # Get the video length
    t=data.shape[1]
   
    # Get the number of pads
    num_pads=int((2*num_slices)-(t%(2*num_slices)))
    #print(num_pads)
    
    # Add padding to the video length
    padded_data=np.pad(data,((0,0),(0,num_pads)),'reflect')
    #print(padded_data[1,370])
    #print(padded_data[1,365:390])
    #print(padded_data.shape)

    return padded_data


# (6) Load autoencoder model

# In[9]:


# Define cost function [mean squared error + correlation loss]
def corr2_mse_loss(a,b):
    a = BK.tf.subtract(a, BK.tf.reduce_mean(a))
    b = BK.tf.subtract(b, BK.tf.reduce_mean(b))
    tmp1 = BK.tf.reduce_sum(BK.tf.multiply(a,a))
    tmp2 = BK.tf.reduce_sum(BK.tf.multiply(b,b))
    tmp3 = BK.tf.sqrt(BK.tf.multiply(tmp1,tmp2))
    tmp4 = BK.tf.reduce_sum(BK.tf.multiply(a,b))
    r = -BK.tf.divide(tmp4,tmp3)
    m=BK.tf.reduce_mean(BK.tf.square(BK.tf.subtract(a, b)))
    rm=BK.tf.add(r,m)
    return rm

# Load autoencoder model
print('Loading autoencoder model...')
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)
model=load_model(autoencoder_path+'autoencoder.h5',custom_objects={'corr2_mse_loss': corr2_mse_loss})
model.load_weights(autoencoder_path+'autoencoder_weights.h5')


# (7)Audio feature extraction 

# In[10]:


# Open the text file of auditory spectrogram input path 
text_file = open(specTxtfile,'r')

# Load each line of the text file
lines = text_file.read().split('\n')
index_shuf=list(range(len(lines)))
lines_shuf=[]
for i in index_shuf:
    lines_shuf.append(lines[i])

# Get the number of audios
num_audios=len(lines)

# Get data shape 
mat=sio.loadmat(lines[0])
data = mat['y'].T[:,2:]

# get_padded_spec function to get the shape after padding data
padded_data=get_padded_spec(data=data)
global AUDIO_LENGTH
AUDIO_LENGTH=padded_data.shape[1]

# Call get_activations function to get the shape after bottleneck features
bottleneck=get_activations(model, 0, 12, padded_data.T)
bottleneck=bottleneck[0].T

# Get the total shape of audio_input variable
audio_input =np.empty((num_audios*(num_slices),bottleneck.shape[0],int(AUDIO_LENGTH/num_slices)), np.dtype('float32'))
print("Audio slices shape:" + str(audio_input.shape))

tmp =np.zeros((AUDIO_LENGTH), np.dtype('float32'))


i=0
for row in lines_shuf:
    
    # Load data from the path
    mat=sio.loadmat(row)
    
    # Read data from the second feature
    data = mat['y'].T[:,2:]
    
    # Call get_padded_spec function: 
    # (1). Compress the spectrogram by raising to the power 1/3 
    # (2). Add padding to the video length 
    padded_data=get_padded_spec(data=data)
   
    
    # Call get_activations function to get the bottleneck feature from autoencoder model 
    # Encoder auditory spectrogram
    bottleneck=get_activations(model, 0, 12, padded_data.T)
    
    #Transpose bottleneck[0] varaible
    bottleneck=bottleneck[0].T
       
    # Call slice_audio_spec function to divide into 15 non-overlap audio slices each of length [390/15] 26
    data=slice_audio_spec(bottleneck)
    #print(data.shape)
    
    #Get the total audio slices
    audio_input[i*(num_slices):(i+1)*(num_slices),:,:]=data[:,:,:]
    i+=1
    if i>=num_audios:
        break
    if i%10==0:
        print(str(i)+'/'+str(num_audios))

audio_output=np.reshape(audio_input,(audio_input.shape[0],audio_input.shape[1]*audio_input.shape[2]))
print('Audio slices shape:'+str(audio_input.shape))
print('Target features to network shape:'+str(audio_output.shape))


# (8) Data_integration

# In[11]:


N= num_train
L=num_slices
print(L)
i=0
for i in range(N):
    if i<(N-1):
        print('Saving data part'+str(i+1)+'...')
        print('Saving test data')
        start=i*L
        end=(i+1)*L
        print(str(start)+' to '+str(end))
        sio.savemat(integrate_data_path+'preprocessed_data_final_part'+str(i+1)+'.mat', mdict={'video_input': video_input[start:end,:,:,:,:], 'audio_input' : audio_input[start:end,:,:]})
    else:
        print('Saving data part'+str(i+1)+'...')
        start=i*L
        end=num_train*num_slices
        print(str(start)+' to '+str(end))
        sio.savemat(integrate_data_path+'preprocessed_data_final_part'+str(i+1)+'.mat', mdict={'video_input': video_input[start:end,:,:,:,:], 'audio_input' : audio_input[start:end,:,:]})

print('Saving validation data...')
start=num_train*num_slices
print(str(start)+' to '+str(video_input.shape[0]))
print(video_input[start:,:,:,:,:].shape)
sio.savemat(integrate_data_path+'preprocessed_data_final_validation.mat', mdict={'video_input': video_input[start:,:,:,:,:], 'audio_input' : audio_input[start:,:,:]})

