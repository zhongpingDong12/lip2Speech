
# coding: utf-8

# In[1]:


import os
import fnmatch
import cv2
import numpy as np
import sys
import glob


# In[4]:


audioPath = 'D:/Mrs_backup/speech_test/audio/s9/'
if not os.path.exists(audioPath):
    os.mkdir(audioPath )
    
for videoPath in glob.glob(r'D:\Mrs_backup\speech_test\video\s9\*.mp4'): 
    print(videoPath)
    
    base=os.path.basename(videoPath)
    fileName= os.path.splitext(base)[0]
    audioFile = audioPath +fileName+'.wav'
    print(audioFile)
    
    os.system('ffmpeg -i '+ videoPath +' -ac 1 -ar 8000 '+audioFile )
        

