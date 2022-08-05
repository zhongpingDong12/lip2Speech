
# coding: utf-8

# In[1]:


def getAudioInputSize(audioSpecPath):
    firstFilePath = audioSpecPath + os.listdir(audioSpecPath)[0]
    audioInputSize=sio.loadmat(firstFilePath)
    return audioInputSize

def interp_Frame(A,D1,D2):
    #print("A.shape IS ",A.shape)
    
    new_dims = []
    for original_length, new_length in zip(A.shape, (D1,D2)):
        new_dims.append(np.linspace(0, original_length-1, new_length))

    coords = np.meshgrid(*new_dims, indexing='ij')
    B = map_coordinates(A, coords)
    print('B.shape IS ',B.shape)
    return B

def getAuidoSpecInput(audio_input,audioSpecPath,audioSpecLength,audioSpecFreq):
    for count, filename in enumerate(sorted(os.listdir(audioSpecPath)),start=0): 
        # Get source (src) addresses
        audioFile  =audioSpecPath+ filename 
        mat=sio.loadmat(audioFile)
        data=np.power(mat['y'],.3)
        
        if(data.shape[0]!=audioSpecLength):
            print(audioSpecPath)
            print(data.shape)
            data= interp_Frame(data,audioSpecLength,audioSpecFreq)
    
        audio_input[count*audioSpecLength:(count+1)*audioSpecLength,:]=data
    return audio_input

def audioSpecClassify(numfiles,audioSpecLength,audio_input,trainRatio):
    numTrain= int(trainRatio * numfiles)
    trainEdge=int(numTrain * audioSpecLength)
    audioInputTrain=audio_input[:trainEdge,:]
    audioInputTest=audio_input[trainEdge:,:]
    return audioInputTrain, audioInputTest

# def audioSpecClassify(numfiles,audioSpecLength,numTrain):
#     trainEdge=int(numTrain * audioSpecLength)
#     audioInputTrain=audio_input[:trainEdge,:]
#     audioInputTest=audio_input[trainEdge:,:]
#     return audioInputTrain, audioInputTest

def createPath(filePath, addFile):
    newFilePath = filePath + addFile
    if not os.path.exists(newFilePath):
        os.mkdir(newFilePath )
    return newFilePath


# In[2]:


# Define cost function
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

def constructModel(adam,reg):
    # Construct autoencode model architecture
    config = BK.tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = BK.tf.Session(config=config)
    model=Sequential()
    
    model.add(Dense(512, input_shape=(audioInputTrain.shape[1],)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(32,kernel_regularizer=l1_l2(.001)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(noise.GaussianNoise(.05))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(audioInputTrain.shape[1]))
    model.compile(loss=corr2_mse_loss, optimizer=adam)
    model.summary()
    return model

def trainModel(model,modelTrainPath,audioInputTrain,audioInputTest,num_iter):
    for i in range(num_iter):
        print('### Autoencoder model, iteration: '+str(i)+'/'+str(num_iter))

        # Train the model
        history = model.fit(audioInputTrain, audioInputTrain, batch_size=128, epochs=1, verbose=1, validation_data=(audioInputTest,audioInputTest))

        # Add the train set loss to the history
        loss_history[i,0]=history.history['loss'][0]

        # Add the validation set loss to the history 
        loss_history[i,1]=history.history['val_loss'][0]

        # Save the loss history
        sio.savemat(modelTrainPath+'loss_history.mat', mdict={'history':loss_history})

        # Save autoencoder model
        model.save(modelTrainPath+'autoencoder.h5')

        # Save the weight of the model
        model.save_weights(modelTrainPath+'autoencoder_weights.h5')
    return loss_history

def plotModelLoss(modelTrainPath,loss_history,num_iter):
    fig = plt.figure()
    plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    loss= loss_history[:,0]
    val_loss= loss_history[:,1]

    plt.plot(loss, 'r-^',label='loss' )
    plt.plot(val_loss, 'b-^',label='val_loss')
    plt.legend(loc='upper right')

    #plt.axis([0, num_iter, -1, -0.9])
    plt.savefig(modelTrainPath+'loss_history.png')  # should before show method
    plt.show()
    
#Define get_activations function
def get_activations(model, layer_in, layer_out, X_batch):
    get_activations = BK.function([model.layers[layer_in].input, BK.learning_phase()], [model.layers[layer_out].output])
    activations = get_activations([X_batch,0])
    return activations

def plotSpectrogram(oriSample,preSample):
    # Plot Original spectrogram
    figure1 = plt.figure(1,figsize=(12, 5))
    plt.imshow(np.power(oriSample,3).T, origin="lower", aspect="auto", interpolation="none")
    plt.title('Original spectrogram')
    
    figure2 = plt.figure(2,figsize=(12, 5))
    plt.imshow(np.power(preSample,3).T, origin="lower", aspect="auto", interpolation="none")
    plt.title('Autoecndoer spectrogram')


# In[3]:


# Import libraries
import fnmatch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.io as sio
import numpy as np
from keras import backend as BK
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape, noise
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, LSTM
from keras.layers.advanced_activations import ELU
from keras.models import Sequential, Model, load_model
from keras.regularizers import l1,l2,l1_l2
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates


# Define input and output path
inputPath ='D:/Mrs_backup/speech_test/AudSpecs/s9/'
outputPath = 'D:/Mrs_backup/speech_test/autoencoder/s9_trainModel/'
if not os.path.exists(outputPath):
    os.mkdir(outputPath )
        
# Load file and get the size
numfiles=len(fnmatch.filter(os.listdir(inputPath), '*.mat'))
print('Total number of video is: ' + str(numfiles))

audioInputSize = getAudioInputSize(inputPath)
audioSpecLength = audioInputSize['y'].shape[0]
audioSpecFreq = audioInputSize['y'].shape[1]
audio_input =np.empty((numfiles*audioSpecLength,audioSpecFreq), np.dtype('float32'))
print("audioSpecLength is: "+ str(audioSpecLength))
print("audioSpecFreq is: "+ str(audioSpecFreq))
print("Audio input shape is: "+ str(audio_input.shape))
print("------------------------------------------ ")

# Collect audio spectrogram data
audio_input = getAuidoSpecInput(audio_input, inputPath,audioSpecLength,audioSpecFreq)
print("Audio example data is:"+ str(audio_input[0,0:10]))
print("------------------------------------------ ")

# Classify audio spectrogram data into training set(80%) and test set (20%)
audioInputTrain, audioInputTest = audioSpecClassify(numfiles,audioSpecLength,audio_input,trainRatio = 0.8)
print('Shape of all the data:'+str(audio_input.shape))
print('Shape of the train data to autoencoder:'+str( audioInputTrain.shape))
print('Shape of the test data to autoencoder:'+str( audioInputTest.shape))
print("------------------------------------------ ")
    
# Construct model
print('-------------- Summary of model -----------------')
model = constructModel(adam=Adam(lr=.0001), reg=.001)
    
# Train autoencoder model and save model output
print('-------------- Train model -----------------')
num_iter=150
loss_history=np.empty((num_iter,2), dtype='float32')
loss_history = trainModel(model,outputPath,audioInputTrain,audioInputTest,num_iter)
    
# predict one sample
oriSample =  audio_input[:audioSpecLength,:audioSpecFreq]
preSample = model.predict(oriSample)
plotSpectrogram(oriSample,preSample)
print('-------------- End of autoencoder-----------------')


# In[4]:


# predict one sample
oriSample =  audio_input[audioSpecLength:audioSpecLength+audioSpecLength,:audioSpecFreq]
preSample = model.predict(oriSample)
plotSpectrogram(oriSample,preSample)
print('-------------- End of autoencoder-----------------')

