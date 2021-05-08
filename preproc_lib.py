from os import listdir
import scipy.io.wavfile as wav
from os.path import isfile, join
import librosa
import librosa.display 
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from AE.spec_lib import pretty_spectrogram
def create_specs(fft, time_long, step_size, log_ref, files_permutation):
  audio_dir='/content/free-spoken-digit-dataset/recordings/'
  file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
  train_list = np.zeros([0,int(fft/2),int(time_long/step_size)])
  test_list = np.zeros([0,int(fft/2),int(time_long/step_size)])

  #sp_sz=2046
  sp_sz = int(time_long)
  i = 0
  for file_name in file_names:
    audio_path = audio_dir + file_name

    sample_rate, samples = wav.read(audio_path)
    samples = np.append(samples, np.random.randn(sp_sz-samples.shape[0]%sp_sz)*10, axis=0)
    ms = np.transpose(pretty_spectrogram(samples.astype("float32"),fft_size=fft,step_size=step_size,log=False))
    n_ms = samples.shape[0]//sp_sz
    ms = np.expand_dims(librosa.power_to_db(ms,
                                            ref=log_ref), axis=0)
    lms = np.split(ms, n_ms, axis=2)
    ms2 = np.concatenate(lms)
    if files_permutation[i]<np.ceil(len(file_names)*0.2):
      test_list = np.append(test_list,ms2,axis=0)   
    else:
      train_list = np.append(train_list,ms2,axis=0)
    i += 1
    return train_list, test_list

class specData():
  def __init__(self,fft,time_long,fft_step_size_ratio,clip=1e0):
    self.fft= fft
    self.time_long = time_long
    self.fft_step_size_ratio = fft_step_size_ratio
    self.clip = clip
  def set_clip(self,clip):
    self.clip = clip
