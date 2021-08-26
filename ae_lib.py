# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:19:12 2021

@author: barberot
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from AE.spec_lib import pretty_spectrogram
from AE.spec_lib import invert_pretty_spectrogram
import librosa
import librosa.display 
import scipy.io.wavfile as wav
from os import listdir
from os.path import isfile, join
from tensorflow import keras
from keras.layers import Activation, Dense, Input, GaussianNoise
from keras.layers import Conv2D, Flatten, BatchNormalization, Dropout
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
from keras import regularizers
np.random.seed(1337)
import time
from keras.models import model_from_json

class myautoencoder():
  def __init__(self,compression_rate,spec, x_train, x_test, mode = 'conv', random_encoder = False, pretrain_encoder = False, encoder_w=None, std = 0, log_ref=1e-5):
    self.mode = mode
    self.time_long = int(spec.time_long)
    #modif
    self.x_train = x_train
    self.x_test = x_test
    self.fft = spec.fft
    self.step_size = int(spec.fft/spec.fft_step_size_ratio)
    self.time = int(self.time_long/self.step_size)
    self.clip = spec.clip
    self.encoder_w = encoder_w
    image_size= self.x_train.shape
    self.compression_rate = compression_rate
    self.x_train = np.reshape(self.x_train, [-1, image_size[1], image_size[2], 1])
    self.x_test = np.reshape(self.x_test, [-1, image_size[1], image_size[2], 1])
    self.x_train = self.x_train.astype('float32') / self.clip
    self.x_test = self.x_test.astype('float32') / self.clip
    self.log_ref = log_ref

    # Network parameters
    input_shape = (image_size[1], image_size[2], 1)
    self.batch_size = 32
    kernel_size = (int(10*self.fft/512),int(5*self.time/16))
    latent_dim =int(self.compression_rate*self.time_long)
    # Encoder/Decoder number of CNN layers and filters per layer
    layer_filters = [16, 32]

    # Build the Autoencoder Model
    # First build the Encoder Model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)
    if mode == 'conv':
      shape = (None,int(self.fft/2), int(self.time_long/self.step_size),int(self.time_long/(self.step_size*2)))
    else:
      shape = (None,int(self.fft/2), int(self.time_long/self.step_size),1)  
    # Generate the latent vector
    x = Flatten()(inputs)

    self.layer = Dense(latent_dim, name='latent_vector')
    latent = self.layer(x)
    latent = K.cast(latent,"int32")
    latent = K.cast(latent,"float32")
    if pretrain_encoder == True:
      self.layer.set_weights(self.encoder_w)
      self.layer.trainable = False
    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    #encoder.summary()
    # Build the Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    if self.mode == 'mlp':
      x = Dense(500, activation="relu")(x) 
      #x = Dropout(0.2)(x)
      x = BatchNormalization()(x)
      x = Dense(500, activation="relu")(x) 
      #x = Dropout(0.2)(x)      
      x = BatchNormalization()(x) 
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    lp = x
    '''
    x = Conv2DTranspose(filters=1,
                        kernel_size=(5,5),
                        padding='same')(x)
    '''
    #x2 = x
    # Stack of Transposed Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use UpSampling2D as alternative to strides>1
    # - faster but not as good as strides>1

    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            kernel_regularizer=regularizers.l2(0.001),
                            strides=(1,1),
                            activation='relu',
                            padding='same')(x)


    conv = Conv2DTranspose(filters=1,
                        kernel_size=kernel_size,
                        padding='same')(x)

    if self.mode == 'conv':
      x2 = conv
    elif self.mode == 'lp' or self.mode == 'mlp':
      x2 = lp 

    outputs = Activation('sigmoid', name='decoder_output')(x2)

    # Instantiate Decoder Model
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()

    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    self.autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
  def train(self, epochs, loss):
    self.epochs = epochs
    self.loss = loss    
    #autoencoder.summary()
    sgd = keras.optimizers.Adam(lr=0.0001, beta_1=0.95, beta_2=0.999, amsgrad=False)
    #sgd = keras.optimizers.RMSprop(lr=0.0001, rho=0.99)
    self.autoencoder.compile(loss=self.loss, optimizer=sgd)

    # Train the autoencoder
    history = self.autoencoder.fit(x = self.x_train,
                    y = self.x_train,
                    validation_data=(self.x_test,self.x_test), verbose=0,
                    epochs=self.epochs,
                    batch_size=self.batch_size)
    if self.mode == 'lp':
      self.encoder_w=self.layer.get_weights()
    # Plot training & validation loss values

    return self.encoder_w, history
  def save_autoencoder(self,files_name):
#g = autoencoder
      g2_json = self.autoencoder.to_json()

      with open(files_name+".json", "w") as json_file:
          json_file.write(g2_json)
# serialize weights to HDF5
      self.autoencoder.save_weights(files_name+".h5")
      print("Saved autoencoder model  to disk")

  def load_autoencoder(self,files_name):
      
      # load json and create model
      json_file = open(files_name+".json", 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      self.autoencoder = model_from_json(loaded_model_json)
      # load weights into new model
      self.autoencoder.load_weights(files_name+".h5")
      print("Loaded model g2 from disk")
  def audio_evaluation(self, num_audios, files_permutation):

    sp_sz = int(self.time_long)
    loss1 = np.zeros(num_audios,)
    loss2 = np.zeros(num_audios,)
    SNR = np.zeros(num_audios,) 
    t0 = time.time()
    total_specs=0 
    audio_dir='/content/free-spoken-digit-dataset/recordings/'
    file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
    for i in range(num_audios):
      
      audio_path = audio_dir + file_names[files_permutation[i]]
      sample_rate, samples = wav.read(audio_path)
      samples = np.append(samples, np.random.randn(sp_sz-samples.shape[0]%sp_sz)*10, axis=0)
      ms = lms = np.transpose(pretty_spectrogram(samples.astype("float32"),fft_size=self.fft,step_size=self.step_size,log=False))
      ms=librosa.power_to_db(ms, ref=self.log_ref)
      n_ms = samples.shape[0]//sp_sz
      total_specs += n_ms
      ms2 = np.expand_dims(ms, axis=0)
      lms = np.split(ms2, n_ms, axis=2)
      ms2 = np.concatenate(lms)
      x_decoded = self.autoencoder.predict(np.reshape(ms2/self.clip, [-1, int(self.fft/2), self.time, 1]))
      lms = np.split(x_decoded, x_decoded.shape[0], axis=0)
      x_decoded2 = np.concatenate(lms,axis=2)
      x_decoded3 = np.reshape(x_decoded2, [ x_decoded2.shape[1], x_decoded2.shape[2]])*self.clip
      comp3 = librosa.core.db_to_power(np.transpose(x_decoded3), ref=self.log_ref)
      recovered_audio_recon2 = invert_pretty_spectrogram(
          comp3, fft_size=self.fft, step_size=self.step_size, log=False, n_iter=20
      )
      b =  pretty_spectrogram(samples.astype("float32")
          ,fft_size=self.fft,step_size=self.step_size,log=False)
      c =  pretty_spectrogram(recovered_audio_recon2.astype("float32")
        ,fft_size=self.fft,step_size=self.step_size,log=False)
      loss1[i]=np.linalg.norm(b-c)/np.linalg.norm(b)
      loss2[i]=np.linalg.norm(b-c)
      SNR[i]=np.linalg.norm(samples)/np.linalg.norm(recovered_audio_recon2[0:samples.shape[0]]-samples)
    loss1_tot = np.sum(loss1)/num_audios
    loss2_tot = np.sum(loss2)/num_audios 
    SNR_tot = np.sum(SNR)/num_audios    
    print('LOSS',loss1_tot)
    print('LOSS2',loss2_tot)
    print('SNR', SNR_tot)
    print(total_specs)    
    print( 'tiempo de reconstrucción: {}s'.  format(int(time.time()-t0)))   
  def make_wav(self,number, person,repetition):
    if person == 1:
      per = '_jackson_'
    elif person == 2: 
      per='_nicolas_'
    elif person == 3: 
      per='_theo_'    
    else:
      per = '_yweweler_'      
    sp_sz = int(self.time_long)
    audio_path='/content/free-spoken-digit-dataset/recordings/'+str(number)+per+str(repetition)+'.wav'
    sample_rate, samples = wav.read(audio_path)
    samples = np.append(samples, np.random.randn(sp_sz-samples.shape[0]%sp_sz)*10, axis=0)
    ms = np.transpose(pretty_spectrogram(samples.astype("float32"),fft_size=self.fft,step_size=self.step_size,log=False))
    ms=librosa.power_to_db(ms,  ref=self.log_ref)
    n_ms = samples.shape[0]//sp_sz
    ms2 = np.expand_dims(ms, axis=0)
    lms = np.split(ms2, n_ms, axis=2)
    ms2 = np.concatenate(lms)
    x_decoded = self.autoencoder.predict(np.reshape(ms2/self.clip, [-1, int(self.fft/2), self.time, 1]))
    lms = np.split(x_decoded, x_decoded.shape[0], axis=0)
    x_decoded2 = np.concatenate(lms,axis=2)
    x_decoded3 = np.reshape(x_decoded2, [ x_decoded2.shape[1], x_decoded2.shape[2]])*self.clip
    comp3 = librosa.core.db_to_power(np.transpose(x_decoded3), ref=self.log_ref)
    recovered_audio_recon2 = invert_pretty_spectrogram(
        comp3, fft_size=self.fft, step_size=self.step_size, log=False, n_iter=20
    )
    return recovered_audio_recon2, sample_rate 
