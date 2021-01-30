from os import listdir
import scipy.io.wavfile as wav
from os.path import isfile, join
import librosa
import librosa.display 
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from spec_lib import pretty_spectrogram
def create_specs(fft, time_long, step_size, log_ref):
  audio_dir='/content/free-spoken-digit-dataset/recordings/'
  file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
  ms_list = np.zeros([0,int(fft/2),int(time_long/step_size)])

  #sp_sz=2046
  sp_sz = int(time_long)
  i = 0
  for file_name in file_names:
    i += 1
    audio_path = audio_dir + file_name

    sample_rate, samples = wav.read(audio_path)
    samples = np.append(samples, np.random.randn(sp_sz-samples.shape[0]%sp_sz)*10, axis=0)
    ms = np.transpose(pretty_spectrogram(samples.astype("float32"),fft_size=fft,step_size=step_size,log=False))
    n_ms = samples.shape[0]//sp_sz
    ms = np.expand_dims(librosa.power_to_db(ms,
                                            ref=log_ref), axis=0)
    lms = np.split(ms, n_ms, axis=2)
    ms2 = np.concatenate(lms)
    ms_list = np.append(ms_list,ms2,axis=0)
  print(np.amin(ms_list))
  print(np.amax(ms_list))  
  return  train_test_split(
    ms_list, test_size=0.20, random_state=42)
class specData():
  def __init__(self,fft,time_long,fft_step_size_ratio,clip=1e0):
    self.fft= fft
    self.time_long = time_long
    self.fft_step_size_ratio = fft_step_size_ratio
    self.clip = clip
  def set_clip(self,clip):
    self.clip = clip
