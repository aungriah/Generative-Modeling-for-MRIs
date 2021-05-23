import numpy as np
from PIL import Image 
from scipy.misc import imsave 




xtrain = np.load('imgs_hist_norm2.npy').astype('float32')
xtrain = np.expand_dims(xtrain,axis=-1)
#eval_xtrain = np.concatenate([xtrain,xtrain,xtrain],axis=-1).transpose(0,3,1,2)

gan = np.load('/Users/aungriah/Documents/ETH/Semester Project/Code/styleGAN/styleGAN.npy').astype('float32')
gan = np.expand_dims(gan,axis=-1)

dfc_vae = np.load('/Users/aungriah/Documents/ETH/Semester Project/Code/DFC_Vae/dfc_vae.npy').astype('float32')
dfc_vae = np.expand_dims(dfc_vae,axis=-1)

introvae = np.load('/Users/aungriah/Documents/introvae/introvae.npy').astype('float32').transpose(0,2,3,1)

idx = np.random.randint(0,1000,100)

gan = xtrain[:130]
dfc_vae = dfc_vae[idx]
introvae = introvae[idx]

gan = np.reshape(gan,(5,26,128,128)).transpose(0,2,1,3)
#originals = np.transpose(originals(0,2,1,3))
gan = np.reshape(gan,(5*128,26*128))

imsave('gan_generations_5.png',gan)

#dfc_vae = np.reshape(dfc_vae,(10,10,128,128)).transpose(0,2,1,3)
#originals = np.transpose(originals(0,2,1,3))
#dfc_vae = np.reshape(dfc_vae,(10*128,10*128))

#imsave('dfcvae_generations.png',dfc_vae)

#introvae = np.reshape(introvae,(10,10,128,128)).transpose(0,2,1,3)
#originals = np.transpose(originals(0,2,1,3))
#introvae = np.reshape(introvae,(10*128,10*128))

#imsave('introvae_generation.png',introvae)

