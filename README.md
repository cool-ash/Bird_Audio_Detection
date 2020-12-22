# Bird_Audio_Detection(https://www.kaggle.com/c/bird-audio-detection):

## Pre-processing: 
<p align="justify">
The utilized Spectro-temporal features are log mel-band energies, extracted from short frames of 10 second audio. These features has been shown to perform well in various
audio tagging, sound analysis and sound detection tasks[1,2,3]. Typical value of amount of FFT points chosen for 44 kHz, is 1024, with 50% overlap(512 fft points) using 
hamming window. Using the approach of 50% overlap helps to obtain much clear spectrum as it take data from centre of signal frames. 40 log mel-band energy features were 
extracted from the magnitude spectrum. Librosa library was used in the feature extraction process.Keeping in mind that human everyday sounds and environment sounds like bird 
singing are often contained in a relatively small portion of the frequency range (mostly around 2-8 kHz), extracting features in that range seems like a good approach while 
deciding for no of mel-bands.
</p>
## Random Mel Spectograms from Training Set(ffbird and wwbird datasets):

![image](https://user-images.githubusercontent.com/42828760/102864316-9ac10680-443c-11eb-8081-6a7e8edc59dd.png)

## Random Mel Spectograms from Test Set:

![image](https://user-images.githubusercontent.com/42828760/102869133-d7dcc700-4443-11eb-9c46-f029e27f14d8.png)


## Model:
<p align="justify">
We first went with a CRNN approach whose inspiration was the Kostas research paper(https://arxiv.org/pdf/1703.02317.pdf). 4 Convolution layers with 96 filters were implemented with relu as activation function, Adam optimizer and binary cross entropy. Two gated recurrent neural net layers were also used at end. The accuracy obtained with this approach after training with both datasets was around 64%. 
</p>
The second model was the simpler model which used CNN, which turned out to better later. 5 CNN layers were used with configurations as followed: 

Layer 1(Input): An input layer to take input of shape (862,40,1). 

Layer 2(CNN): Kernel Size: (7,7), No of filters: 8, activation-tanh. 

BatchNormalization() 

MaxPooling2D(2,2) 

Layer 3(CNN): Kernel Size (5,5) , No of filters: 16, activation-tanh. 

MaxPooling2D(2,2) 

Dropout(0.4) 

Layer 4(CNN): Kernel Size (5,5) , No of filters: 32, activation-tanh. 

MaxPooling2D(2,2) 

Dropout(0.4) 

Layer 5(CNN): Kernel Size (5,5) , No of filters: 32, activation-tanh. 

MaxPooling2D(2,2) 

Dropout(0.4) 

Layer 6(CNN): Kernel Size (5,5) , No of filters: 64, activation-tanh , L1(0.001) and L2(0.001) Regularization. 

MaxPooling2D(2,2) 

Dropout(0.4) 

Flatten() 

Layer 7(Dense): No of Neurons: 64, activation-tanh, L2(0.001) Regularization.  

Layer 8(Output Layer): 1 Neuron, activation=sigmoid. 

 

Model layer short description: 
<p align="justify">
Conv2D Layer: Use of the Conv2D layer is for the creation of filters for feature extraction. Each layer used different set of filters and kernel sizes to create feature maps as mentioned in model configuration. tanh was used as activation function which helped to learn non-linear relationships present in the spectrogram, tough relu is regarded better but tanh worked better here for out dataset. 
</p>
<p align="justify">
Batch Normalization Layer: Increases the training speed by evenly wide spreading the data in a smaller range and decreasing initial weights importance.  It also takes care of the probability distribution of input images and reduces the effect of the covariance shift.  
</p>
<p align="justify">
L1&L2 Regularization: L1 regularization helps us avoid underfitting and feature selection by removing or reducing the less important features to zero. L1 regularization employs Lasso Regression adds absolute value of magnitude of coefficient as penalty term to the loss function. L2 regularization also helped in underfitting and overfitting of our model by employing a technique called Rigid regression, it adds squared magnitude of coefficient as penalty term to the loss function.We  sum  both the penalties and achieved a good results as its removed the less import features and also avoided over and underfit of the model.  
</p>
MaxPooling2D Layer: A window of 2X2 size with stride 1 is used to extract the most prominent features from feature maps.  

Dropout Layer: Dropout layer has been used in the model to avoid overfitting/learning.  

Flatten Layer: It converts the dimensional inputs into a vector to be able get processed by neural network.  

<p align="justify">
Dense Layer: The penultimate layer contains 64 neurons with tanh activation and last layer of the model defines the number of outputs required (in our case 1) with activation as sigmoid, which gives the probability of bird prediction. 
</p>
Compile Parameters: 

Optimizer – Adam: It providers faster (computationally) and steady gradient decent (initially slow and picks up overtime with every epochs and parameter) without affecting the learning rate.  

Loss – binaryCrossentropy: This loss is used when we have class of either true or false. 

Metrics – Accuracy: As we are trying to predict a class of sample in form of accuracy. 
<p align="justify">
Training: Initially training was done using 2-fold cross validation technique. We observed that the training and validation accuracy reached close to  91% and 83-85% respectively. With this we achieved a test score of 0.57 in public leaderboard. With little bit of tweaking with kernel size, dropouts and adding l1 and l2 penalties, score increased significantly to 0.697 score. At end to get better results we trained the model with both dataset and achieved a score of 0.7068 on test samples(public) with 315 epochs.  
</p>

## References:
<p align="justify">
[1] E. Cakir, T. Heittola, H. Huttunen, and T. Virtanen, “Polyphonic sound event detection using multi-label deep neural networks,” in IEEE International Joint Conference on Neural Networks (IJCNN), 2015.
[2] “Detection and classification of acoustic scenes and events (DCASE),” 2016. [Online]. Available:http://www.cs.tut.fi/sgn/arg/dcase2016/tasksound-event-detection-in-real-life-audio.
[3] E. Cakir, G. Parascandolo, T. Heittola, H. Huttunen, and T. Virtanen, “Convolutional recurrent neural networks for polyphonic sound event detection,” in IEEE/ACM TASLP Special Issue on Sound Scene and Event Analysis, 2017, accepted for publication.
</p>

