
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import gc
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, LSTM
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras import backend as BK
from keras.regularizers import l2
from sklearn import preprocessing
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
import cv2
import IPython
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from scipy import signal
from scipy.signal import firls, lfilter
from numpy.lib import stride_tricks
import scipy.io as sio


# In[2]:


def corr2_loss(a,b):
    a = BK.tf.subtract(a, BK.tf.reduce_mean(a))
    b = BK.tf.subtract(b, BK.tf.reduce_mean(b))
    tmp1 = BK.tf.reduce_sum(BK.tf.multiply(a,a))
    tmp2 = BK.tf.reduce_sum(BK.tf.multiply(b,b))
    tmp3 = BK.tf.sqrt(BK.tf.multiply(tmp1,tmp2))
    tmp4 = BK.tf.reduce_sum(BK.tf.multiply(a,b))
    r = -BK.tf.divide(tmp4,tmp3)
    return r

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


# In[3]:


NUM_TRAIN=600

word='color'
testword='blue'
data_input = 'C:/Users/Andrew/Desktop/word/14_gp_multInput/'+word+'/'
autoencoder_path='C:/Users/Andrew/Desktop/word/04_autoencoder/'+testword+'/'
sample='C:/Users/Andrew/Desktop/word/sample/'
model_ouput = 'C:/Users/Andrew/Desktop/word/16_gp_multOutput/'+word+'/'
if not os.path.exists(model_ouput):
    os.mkdir(model_ouput)


mat1=sio.loadmat(data_input+'preprocessed_data_final_part1.mat')
#video_input_shape = mat1['video_input'].shape
video_input_shape = mat1['video_input'].shape
audio_input_shape= mat1['audio_input'].shape
print('video_input_shape '+str(video_input_shape))
del mat1

mat2=sio.loadmat(data_input+'preprocessed_data_final_part'+str(NUM_TRAIN)+'.mat')
video_input_shape_last=mat2['video_input'].shape
epoch=32.0
steps_per_epoch=int(np.ceil(video_input_shape[0]/epoch)*(NUM_TRAIN-1)+np.ceil(video_input_shape_last[0]/epoch))
print('steps_per_epoch = '+str(steps_per_epoch))
del mat2

gc.collect()


# In[4]:


def data_augmentation(video):
    augmentation_type=[1,2,3]
    video=np.transpose(video,axes=[2,3,1,4,0])
    for i in range(video.shape[4]):
        a_type=np.random.choice(augmentation_type)
        #print('Augmenting data type:'+str(a_type))
        if a_type==1: #Do the flip
            video[:,:,:,:,i]=np.fliplr(video[:,:,:,:,i])
            continue
        if a_type==2: #Do the noise
            video[:,:,:,:,i]+=np.random.normal(0,.01,(video[:,:,:,:,i].shape))
            continue
        if a_type==3:
            continue #Return the original frame
    video=np.transpose(video,axes=[4,2,0,1,3])
    return video

def generate_train_data():
    for j in range(1,NUM_TRAIN+1):
        mat_tmp=sio.loadmat(data_input+'preprocessed_data_final_part'+str(j)+'.mat')
#         video_input = mat_tmp['video_input']
        video_input = mat_tmp['video_input']
        audio_output = mat_tmp['audio_input']
        
        audio_output=np.reshape(audio_output,(audio_output.shape[0],audio_input_shape[1]*audio_input_shape[2]))
        augmented_vid=data_augmentation(video_input)
        yield (augmented_vid,audio_output)


# In[5]:


mat=sio.loadmat(data_input+'preprocessed_data_final_validation.mat')
video_input_test=mat['video_input']
audio_input_test=np.reshape(mat['audio_input'],(-1,audio_input_shape[1]*audio_input_shape[2]))
nb_v=video_input_test.shape[0]
#Dividing validation data to %50 validation during the training and %50 test for completely unseen data
nb_half=int(np.floor(nb_v/2))
video_input_validation=video_input_test[:nb_half,:]
audio_input_validation=audio_input_test[:nb_half,:]
video_input_test=video_input_test[nb_half:,:]
audio_input_test=audio_input_test[nb_half:,:]
mat=None


# In[6]:


config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)
adam=Adam(lr=.0001)
reg=.0005
model=Sequential()


# In[7]:


# 1st layer
model.add(Convolution3D(filters = 16, kernel_size=[3, 3, 3], input_shape=video_input_shape[1:], data_format='channels_first', kernel_initializer="he_normal",padding='same', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 1), data_format='channels_first'))
model.add(Dropout(.25))

# Reshaping spatio-temporal features to feed into LSTM layer
shape=model.get_output_shape_at(0)
model.add(Reshape((shape[-1],shape[1]*shape[2]*shape[3])))


# LSTM layer
model.add(LSTM(512, return_sequences=True, kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(.25))

# 1st LSTM layer
model.add(LSTM(512,input_shape=video_input_shape[1:],return_sequences=True, kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(.25))

# Flattening the output
model.add(Flatten())

# 1st dense layer
model.add(Dense(2048,kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(.4))

# Output layer
model.add(Dense(audio_input_shape[1]*audio_input_shape[2],kernel_initializer="he_normal",use_bias=True))
model.add(Activation('sigmoid'))

model.compile(loss=corr2_mse_loss,optimizer=adam)
model.summary()


# In[8]:


num_iter=50
sio.savemat(model_ouput+'main_encoded_test.mat', mdict={'encode': np.reshape(audio_input_test,(audio_input_test.shape[0],audio_input_shape[1],audio_input_shape[2]))})
predict_final = np.empty((num_iter,audio_input_test.shape[0],audio_input_shape[1],audio_input_shape[2]), dtype='float32')
filepath=model_ouput+"Best_weights_LipReading.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
loss_history=np.empty((num_iter,2), dtype='float32')
for i in range(num_iter):
    print('################ LipReading model, iteration: '+str(i)+'/'+str(num_iter))
    history = model.fit_generator(generator=generate_train_data(),steps_per_epoch=steps_per_epoch, callbacks=callbacks_list,  validation_data=(video_input_validation,audio_input_validation), epochs=1, verbose=1, max_q_size=10)
    predict = model.predict(video_input_test)
    predict = np.reshape(predict,(predict.shape[0],audio_input_shape[1],audio_input_shape[2]))
    predict_final[i,:,:,:] = predict
    loss_history[i,0]=history.history['loss'][0]
    loss_history[i,1]=history.history['val_loss'][0]
    if i>3:
        if loss_history[i-4,1]<loss_history[i,1] and loss_history[i-4,1]<loss_history[i-1,1] and loss_history[i-4,1]<loss_history[i-2,1] and loss_history[i-4,1]<loss_history[i-3,1]:
            print("########### Loss didn't improve after 4 epochs, lr is divided by 5 ############")
            BK.set_value(model.optimizer.lr, .2*BK.get_value(model.optimizer.lr))

    sio.savemat(model_ouput+'predict_encoded_test.mat', mdict={'encode': predict_final,'history':loss_history})
    if i%10==0:
        model.save(model_ouput+'model_LipReading.h5')
        model.save_weights(model_ouput+'LipReading_mid_weights.h5')


# In[9]:


print(loss_history)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.title('corr2_mse_loss and iteartions')
plt.ylabel('corr2_mse_loss')
plt.xlabel('number of iteration')

loss= loss_history[:,0]
val_loss= loss_history[:,1]
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
         
plt.plot(loss, 'r^',val_loss,'b^')
         
plt.axis([0, num_iter, -1, 0])
fig.savefig(model_ouput+'loss_history.png', dpi=fig.dpi)
plt.show()


# In[10]:


#Define get_activations function
def get_activations(model, layer_in, layer_out, X_batch):
    get_activations = BK.function([model.layers[layer_in].input, BK.learning_phase()], [model.layers[layer_out].output])
    activations = get_activations([X_batch,0])
    return activations


# In[11]:


autoencoder=load_model(autoencoder_path +'autoencoder.h5',custom_objects={'corr2_mse_loss': corr2_mse_loss})
autoencoder.load_weights(autoencoder_path +'autoencoder_weights.h5')


# In[12]:


mat1=sio.loadmat(data_input+'preprocessed_data_final_part2.mat')
video_input = mat1['video_input']
audio_input = mat1['audio_input']
model_output = model.predict(video_input)
print(model_output.shape)
model_output_reshape1=np.reshape(model_output,(audio_input.shape[0],audio_input.shape[1],audio_input.shape[2]))
model_output__reshape2=np.concatenate(model_output_reshape1,axis=1)
model_output_final = model_output__reshape2.T
print(model_output_reshape1.shape)
print(model_output_final.shape)


# In[14]:


#orig_data_path='C:/Users/sally/Desktop/MRes_backup/data_set/gabor_features/s1_please/audio/AudSpecs/000.mat'
original_data=sio.loadmat(sample+testword+'.mat')
original_data= original_data['y']
#print(original_data_shape)

# Plot Original spectrogram
figure1 = plt.figure(1,figsize=(20, 3))
plt.imshow(original_data.T, origin="lower", aspect="auto", interpolation="none")
plt.title('Original spectrogram')
figure1.savefig(model_ouput+'Original spectrogram.png', dpi=figure1.dpi)

  
decoder_output =get_activations(autoencoder, 13, 19, model_output_final)[0]
print(decoder_output.shape)
final_model_output = np.power(decoder_output,3)
print(final_model_output.shape)
sio.savemat('final_model_output.mat', mdict={'final_model_output': final_model_output})
    
# Plot Reconstruted spectrogram after reserving the compression process
figure2 = plt.figure(2,figsize=(20, 3))
plt.imshow(final_model_output.T, origin="lower", aspect="auto", interpolation="none")
plt.title('Reconstruted spectrogram')
figure2.savefig(model_ouput+'Reconstruted spectrogram.png', dpi=figure1.dpi)

