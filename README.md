# Bird_Audio_Detection:

## Pre-processing:
The utilized Spectro-temporal features are log mel-band energies, extracted from short frames of 10 second audio. These features has been shown to perform well in various audio tagging, sound analysis and sound detection tasks [1,2,3]. Typical value of amount of FFT points chosen for 44 kHz, is 1024, with 50% overlap(512 fft points) using hamming window. Using the approach of 50% overlap helps to obtain much clear spectrum as it take data from centre of signal frames. 40 log mel-band energy features were extracted from the magnitude spectrum. Librosa library was used in the feature extraction process.Keeping in mind that human everyday sounds and environment sounds like bird singing are often contained in a relatively small portion of the frequency range (mostly around 2-8 kHz), extracting features in that range seems like a good approach which deciding for no of mel-bands.

## Random Mel Spectograms from Training Set(ffbird and wwbird datasets):

![image](https://user-images.githubusercontent.com/42828760/102864316-9ac10680-443c-11eb-8081-6a7e8edc59dd.png)

