from keras.models import model_from_json
from scipy.misc import imsave

from gan2 import *

def laplacian_variance(images):

	#images = decoder.predict(noise)[:,:,:,0]
	return [cv2.Laplacian(image*255, cv2.CV_32F).var() for image in images]

def recon_acc():

	encoded = encoder.predict(trials)
	decoded = decoder.predict(encoded)
	mse = np.mean(np.square(trials[:,:,:,0]-decoded[:,:,:,0]))
	return mse


xtrain = np.load('/Users/aungriah/Documents/ETH/Semester Project/GoodData/normal_brians.npy').astype('float32')
xtrain = np.expand_dims(xtrain,axis=-1)
Images = xtrain 

file = open("network/gen.json", 'r')
json = file.read()
file.close()
gen = model_from_json(json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
gen.load_weights("network/gen.h5")

random_ims = gen.predict([noise(1000),noiseImage(1000),np.ones([1000,1])])[:,:,:,0]

imsave('_.png',random_ims[0])
print(random_ims.shape)
np.save('styleGAN.npy',random_ims)


