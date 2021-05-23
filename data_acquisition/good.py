import numpy as np 
from scipy.misc import imsave 
from matplotlib.image import imread
import importDicoms as iD 

'''
im1 = '../DFC_VAE/model6/images_6/generation_600.png'
im2 = '../DFC_VAE/model6/images_6/reconstruction_600.png'

im3 = '../DFC_VAE/model7/images_7/generation_450.png'
im4 = '../DFC_VAE/model7/images_7/reconstruction_450.png'

im1 = imread(im1)
im2 = imread(im2)
im3 = imread(im3)
im4 = imread(im4)
print(np.amax(im1))
print(np.amax(im2))
print(np.amax(im3))
print(np.amax(im4))

imsave('gen600.png',im1*255)
imsave('recon600.png',im2*255)
imsave('gen450.png',im3*255)
imsave('recon450.png',im4*255)
'''

pathtoDicom = 'Users/aungriah/Documents/ETH/Semester Project/Data/20190502_trainAndValid/ims/0'
path = 'Users/aungriah/Documents/ETH/Semester Project/Data/20190502_databaseNormalMR'
all_cases = iD.loadDicomFromFolderIn3D(pathtoDicom)

np.save('all_cases.npy',all_cases)

print(all_cases.shape)
