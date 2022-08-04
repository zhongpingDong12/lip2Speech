# Lip2Speech
Generate intelligent speech from lip Gabor features

## Abstract
In this study, we proposed two advanced speech reconstruction systems based on the fast and lightweight Gabor extraction method: **GaborPic2Speech** and **GaborFea2Speech**. 

Our GaborPic2Speech system is an end-to-end neural network. It applied Gabor filter technology filtered out irrelevant facial and lip information, and use the most relevant lip pixels as visual input data. Compared to baseline Lip2AudSpec system, which used 7 layers CNN to learn the entire face image, GaborPic2Speech reduced the size of the input data, simplifies the model structure, and improves the accuracy to a certain extent.

To eliminate the drawbacks of unexplainable and complexity of CNNs in visual learning process, we applied Gabor extracted technology and designed a sequence-to-sequence learning (Seq2Seq) speech reconstruction system: GaborFea2Speech. It used the accurate Lip Gabor features as visual input and entirely removed the CNN layer for visual learning. Compared with the Lip2AudSpec and GaborPic2Speech systems, it significantly reduces the input size and simplifies the model structure and at the same time improves the model explainability. Moreover, GaborFea2Speech system, as a Seq2Seq learning, directly maps a sequence of visual Gabor features to the auditory spectrogram, which achieves superior speech reconstruction results in all the scenarios, especially in the multiple-speaker test. 

We used the Grid Corpus dataset to test the results on single speaker, multiple speakers, and vocabularies.  Our experiment results show that our GaborPic2Speech and GaborFea2Speech can reconstruct the original auditory spectrogram from dynamic lip features. On single-speaker test, it achieves 73% and 72% accuracies, respectively, outperforming the 63% accuracy of the baseline Lip2AudSpec. In addition, our experiments also validate the excellent vocabulary reconstruction ability of the GaborPic2Speech and GaborFea2Speech systems, with average accuracies of 74% and 81%, respectively. 

Moreover, our GaborFea2Speech system can also reconstruct intelligent speech in multi-speaker scenarios. As the number of speakers increases, this system can still maintain robust speech construction capability (1-S, 2-S, 4-S, 6-S, 8-S; 72%, 68%, 71%, 65%, 61%), which is incomparable with other speech reconstruction systems, for example, GaborPic2Speech (1-S, 2-S, 4-S, 6-S, 8-S; 73%, 72%, 56%, 54%, 50%) and Lip2AudSpec (1-S, 2-S, 4-S, 6-S, 8-S; 0.63%, 40%, 39%, 33%, 29 %). This breaks the limitation that existing speech reconstruction systems can be only used in single-speaker scenarios, which broadens the practical applications of speech reconstruction systems in multi-speaker scenarios.

**You can find all results analyisis details: _analysis_**

## Architecture
We developed two Gabor based speech recontruction system: GaborPic2Speech and GaborFea2Speech and we uses the deep end-to-end Lip2AudSpec as the basline system.

Our GaborPic2Speech system is an end-to-end neural network. It uses the lip Gabor Picture as visual input, and fed into 1-layer convolutional neural network (CNN) for image learning and 1-layer Long short-term memory (LSTM) neural network for sequence learning, and then flatten to dense layer to generate the output.  Then, it decoded the output to generate auditory spectrogram and transfer to audio waveform. 

Our GaborFea2Speech system is a Sequence-to-sequence learning (Seq2Seq). It uses seven Gabor features values extracted from dynamic lip frames as visual input, and fed into 1-layer Long short-term memory (LSTM) neural network for sequence learning, and then flatten to dense layer to generate the output.
 
![Main Network](figures/Network_main.png)

## Requirements
We implemented the code in python3 using tensorflow, keras, scipy, dlib, skimage, pandas, numpy, cv2, sklearn, IPython, fnmatch. The mentioned libraries should be installed before running the codes. All the libraries can be easily installed using pip:
```shell
pip install tensorflow-gpu keras scipy opencv-python sklearn
```
The backend for Keras can be changed easily if needed.

## Data preparation
This study is based on GRID corpus(http://spandh.dcs.shef.ac.uk/gridcorpus/). To run the codes, you need to first download and preprocess both videos and audios.

By running **_Gabor_extraction.py_** frames will be cropped from videos and Lip Gabor Pictures and Gabor Features will be extracted.

In order to generate auditory spectrograms, the audios should be processed by NSLTools(http://www.isr.umd.edu/Labs/NSL/Software.htm) using **_wav2aud_** function in Matlab.

After extarcted visual and audio data, we then window and integrate all data  in **_.mat_** formats for model training. This can be done by running **_modelInput_collection.py_**

### Training the models
Once data preparation steps are done, autoencoder model could be trained on the auditory spectrograms corresponding to valid videos using **_train_autoencoder.py_**. Training the main network could be performed using **_train_GP2Speech.py_** and **_train_GF2Speech.py_**


## Demo

You can find all demo files here. **_demo_**.

A few samples of the network output are given below:

 **Demo on Single Speaker**     | **Demo on Multiple Speakers**      | **Demo on Vocabulary**     
 ------------- | ------------- | -------- 
 [![Sample1](https://img.youtube.com/vi/-apenOxMQM8/0.jpg)](https://youtu.be/-apenOxMQM8)         | [![Sample2](https://img.youtube.com/vi/62pQrLAbw8E/0.jpg)](https://youtu.be/62pQrLAbw8E)       |[![Sample3](https://img.youtube.com/vi/yv0-dakuY6k/0.jpg)](https://youtu.be/yv0-dakuY6k)  



## Cite
The related work, you can cite:
```
@inproceedings{abel2018fast,
  title={Fast lip feature extraction using psychologically motivated gabor features},
  author={Abel, Andrew and Gao, Chengxiang and Smith, Leslie and Watt, Roger and Hussain, Amir},
  booktitle={2018 IEEE Symposium Series on Computational Intelligence (SSCI)},
  pages={1033--1040},
  year={2018},
  organization={IEEE}
}
```
```
@article{akbari2017lip2audspec,
  title={Lip2AudSpec: Speech reconstruction from silent lip movements video},
  author={Akbari, Hassan and Arora, Himani and Cao, Liangliang and Mesgarani, Nima},
  journal={arXiv preprint arXiv:1710.09798},
  year={2017}
}
```
