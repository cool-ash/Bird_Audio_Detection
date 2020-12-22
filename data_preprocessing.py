import librosa as lb
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, models, layers
from tqdm import tqdm
import keras
import matplotlib.pyplot as plt
import librosa.display
import cv2
import sys

data_wa = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/warblrb10k_public_metadata_2018.csv')                  #In similar manner do for ffbird data
xwa_ml50 = []
for i in tqdm(range(len(data_wa))):
  audio, fs = lb.load('wawav'+'/'+str(data_wa['itemid'][i])+'.wav', sr = 44100)
  if len(audio)>441000:
     audio = audio[:441000]
  elif len(audio)<441000:
    if 441000-len(audio) < len(audio):
      audio = np.concatenate((audio,audio[:441000-len(audio)]))
    else:
      temp = np.zeros(441000)
      temp[:len(audio)] = audio
      audio = temp
  audio = audio * 1/np.max(np.abs(audio))
  kwargs_for_mel = {'n_mels': 40}
  ml =  np.log(lb.feature.melspectrogram(y=audio, sr=fs, n_fft=1024, hop_length=512, power=1, **kwargs_for_mel)+sys.float_info.epsilon)
  xwa_ml50.append(ml)
np.save('xwa_ml50.npy',np.asarray(xwa_ml50))
