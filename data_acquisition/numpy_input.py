import tensorflow
import numpy as np 
from keras.datasets import mnist
from PIL import Image
from scipy.misc import imsave
'''
x_train = np.load('cropped_ims.npy')


pics = np.zeros(x_train.shape)

for i in range(x_train.shape[0]):
	if (i%100==0):
		print(i)
	photo = imsave('train_images/im_{}.png'.format(i),x_train[i])


for i in range(x_train.shape[0]):
	if (i%100==0):
		print(i)
	im = Image.open('train_images/im_{}.png'.format(i))
	pics[i,:,:]=im

np.save('imgs.npy',pics)
'''
array = np.load('imgs.npy')

print(np.amax(array))

