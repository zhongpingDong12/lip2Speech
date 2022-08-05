
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


# In[2]:


speaker = 's9'

gabor_path = 'D:/Mrs_backup/speech_test/gabor_feature/'+ speaker + '/'
audio_path = 'D:/Mrs_backup/speech_test/AudSpecs/'+ speaker + '/'
autoencoder_path= 'D:/Mrs_backup/speech_test/autoencoder/'+ speaker + '_trainModel/'
integrate_data_path ='D:/Mrs_backup/speech_test/model_input/gf/'+ speaker + '/'
if not os.path.exists(integrate_data_path):
    os.mkdir(integrate_data_path)
    
print('gabor input path is '+ gabor_path)
print('audio path is '+ audio_path)
print('autoencoder path is: '+autoencoder_path)
print('integrate path is: '+integrate_data_path)

num_files = len(os.listdir(gabor_path))
print(num_files)
num_slices=15
slice_length=26
num_features=7
num_frame =num_slices*slice_length
time_diff =3


# In[3]:


# *****************************  Interplate the length of frame equals to the video length *************************#
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

def interp_Frame(A):
    #print("A.shape IS ",A.shape)
    
    new_dims = []
    for original_length, new_length in zip(A.shape, (num_frame,num_features)):
        new_dims.append(np.linspace(0, original_length-1, new_length))

    coords = np.meshgrid(*new_dims, indexing='ij')
    B = map_coordinates(A, coords)
    #print('B.shape IS ',B.shape)
    return B

# *****************************  Define diff function *************************#
def diff(buf_input):
    buf_input=np.pad(buf_input,((1,0),(0,0)),'edge')
    buf_output=np.diff(buf_input,axis=0)
    return buf_output


# ********************** slice_video_3D function to divide into 15 non-overlap slice  **************************#
def slice_video_3D(video):
    video_output =np.empty((num_slices,slice_length,num_features*time_diff), np.dtype('float32'))
    start=0
    #print(video.shape)
    for i in range(0,num_slices):
        video_output[i,:,:]=video[start:start+slice_length,:]
        start+=slice_length
    return video_output


# In[4]:


video_input =np.empty((num_files*(num_slices),int(slice_length),int(num_features)*time_diff), np.dtype('float32'))
print(video_input.shape)


# In[5]:


j=0
for filename in os.listdir(gabor_path): 
    src =gabor_path+ filename 
    print(src)
    format_num1="{number:03}".format(number=j)
    dst =gabor_path + str(format_num1)+'.mat'
    print(dst)
    os.rename(src, dst)
    j=j+1   


# In[6]:


start = 0
for i in range(num_files):
#for i in range(1):

    #------------------- Get data path ------------------------------ *
    format_i="{number:03}".format(number=i)
    data_path = gabor_path + format_i+ '.mat'
    print(data_path)
    
    #------------------- Load data from data path  ------------------------------ *
    gabor_data = sio.loadmat(data_path)
    gabor_data = gabor_data['gabor_input']
    
    
    #------------- Call interplation function to interplate the length of frame equals to video length ------------ *
    gabor_data = interp_Frame(gabor_data)


     # ---------------------  Call diff function --------------------------------*
    diff_video=np.empty((gabor_data.shape[0],num_features*time_diff))
    diff_video[:,0:num_features*1]=gabor_data
    diff_video[:,num_features:num_features*2]=diff(gabor_data)
    diff_video[:,num_features*2:num_features*3] =diff(diff_video[:,num_features:num_features*2])
    #print(diff_video.shape)
    
    #------------------------- Call slice_video_3D function ----------------------*
    data_vid=slice_video_3D(diff_video)
    
    # --------------------- Add total number of slices ---------------------------*
    video_input[start:start+num_slices,:,:]=data_vid
    start+=num_slices
    
    
print('Video slices shape:'+str(video_input.shape))


# In[7]:


print(video_input.shape)
print(video_input[200,0:3,:])
print("------------------")


# (4) Audio features

# In[8]:


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

# Define slice_audio_spec function to to divide into 15 non-overlap slices each of length 26
def slice_audio_spec(audio_spec):
    global AUDIO_LENGTH
    window_size=int(AUDIO_LENGTH/num_slices) #from time to number of audio index 
    audio_output =np.empty((num_slices,audio_spec.shape[0],window_size), np.dtype('float32'))
    start=0
    for i in range(0,num_slices):
        audio_output[i,:,:]=audio_spec[:,start:start+window_size]
        start+=window_size
        if start>AUDIO_LENGTH-window_size:
            break
    return audio_output

# Define get activations function # Extract the 32-bin bottleneck features as target for the main network
def get_activations(model, layer_in, layer_out, X_batch):
    get_activations = BK.function([model.layers[layer_in].input, BK.learning_phase()], [model.layers[layer_out].output])
    activations = get_activations([X_batch,0])
    return activations

#Define padding function
def get_padded_spec(data): 
    # Compress the spectrogram by raising to the power 1/3
    data=np.power(data,.3)
    # Get the video length
    t=data.shape[1]
    # Get the number of pads
    num_pads=int((2*num_slices)-(t%(2*num_slices)))
    # Add padding to the video length
    padded_data=np.pad(data,((0,0),(0,num_pads)),'reflect')
    #print(padded_data.shape)
    return padded_data


# In[9]:


# Load autoencoder model
print('Loading autoencoder model...')
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)
model=load_model(autoencoder_path+'autoencoder.h5',custom_objects={'corr2_mse_loss': corr2_mse_loss})
model.load_weights(autoencoder_path+'autoencoder_weights.h5')


# In[10]:


j=0
for filename in os.listdir(audio_path): 
    src =audio_path+ filename 
    print(src)
    format_num1="{number:03}".format(number=j)
    dst =audio_path + str(format_num1)+'.mat'
    print(dst)
    os.rename(src, dst)
    j=j+1   


# In[11]:


import fnmatch
audioSpec = 'D:/Mrs_backup/speech_test/AudSpecs/'
audio_path = 'D:/Mrs_backup/speech_test/AudSpecs/'+ speaker + '/'

#Create a text file for audio input path
file = open(audioSpec +'s9_valid_aud_specs.txt','w')   
file.close()

# Get the amount of files in face folder [total 10 videos in the folders]
numfiles=len(fnmatch.filter(os.listdir(audio_path), '*.mat'))
print(numfiles)

for j in range(0,numfiles):
    format_num1="{number:03}".format(number=j)
    #  Write path to audio input file
    with open(audioSpec +'s9_valid_aud_specs.txt', 'a') as file:
        file.write(audio_path + str(format_num1)+'.mat\n')


# In[12]:



# Open the text file of auditory spectrogram input path 
text_file = open(audioSpec+'s9_valid_aud_specs.txt','r')

# Load each line of the text file
lines = text_file.read().split('\n')
index_shuf=list(range(len(lines)))
lines_shuf=[]
for i in index_shuf:
    lines_shuf.append(lines[i])

# Get the number of audios
num_audios=len(lines)
print(num_audios)

# Get data shape 
mat=sio.loadmat(lines[0])
data = mat['y'].T[:,2:]

# get_padded_spec function to get the shape after padding data
padded_data=get_padded_spec(data=data)
global AUDIO_LENGTH
AUDIO_LENGTH=padded_data.shape[1]
# global AUDIO_LENGTH
# AUDIO_LENGTH= 182
#print(AUDIO_LENGTH)

# Call get_activations function to get the shape after bottleneck features
bottleneck=get_activations(model, 0, 12, padded_data.T)
#bottleneck=get_activations(model, 0, 15, padded_data.T)
#bottleneck=get_activations(model, 0, 18, padded_data.T)
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
    #print(padded_data.shape)
    
    #padded_data=interp_Spectrog(padded_data) 
    #print(padded_data)
    # Call get_activations function to get the bottleneck feature from autoencoder model 
    # Encoder auditory spectrogram
    bottleneck=get_activations(model, 0, 12, padded_data.T)
    #bottleneck=get_activations(model, 0, 15, padded_data.T)
    #bottleneck=get_activations(model, 0, 18, padded_data.T)
    
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


# In[13]:


print(audio_input[500,:,0])
print(audio_output[500,])


# In[14]:


print(video_input.shape[1])
print(audio_input.shape)


# In[15]:


N=80
num_test=20
num_train=num_audios-num_test
L=int(np.ceil(num_train/N)*num_slices)
i=0
for i in range(N):
    if i<79:
        print('Saving data part'+str(i+1)+'...')
        print('Saving test data')
        start=i*L
        end=(i+1)*L
        print(str(start)+' to '+str(end))
        sio.savemat(integrate_data_path+'preprocessed_data_final_part'+str(i+1)+'.mat', mdict={'video_input': video_input[start:end,:,:], 'audio_input' : audio_input[start:end,:,:]})
    else:
        print('Saving data part'+str(i+1)+'...')
        start=i*L
        end=num_train*num_slices
        print(str(start)+' to '+str(end))
        sio.savemat(integrate_data_path+'preprocessed_data_final_part'+str(i+1)+'.mat', mdict={'video_input': video_input[start:end,:,:], 'audio_input' : audio_input[start:end,:,:]})

print('Saving validation data...')
start=num_train*num_slices
print(str(start)+' to '+str(video_input.shape[0]))
sio.savemat(integrate_data_path+'preprocessed_data_final_validation.mat', mdict={'video_input': video_input[start:,:,:], 'audio_input' : audio_input[start:,:,:]})

