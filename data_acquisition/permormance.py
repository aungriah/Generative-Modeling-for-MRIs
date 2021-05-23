import numpy as np 
import tensorflow as tf 
import keras 
import cv2
import keras
#from keras.applications import VGG16
from keras.activations import sigmoid
from keras import backend as K 
from keras.models import Model, Input, load_model
from keras.layers import *
from keras.optimizers import Adam 
from vgg16 import *

import matplotlib.pyplot as plt 



###############################################################################################################
# This script is inteded to evaluate the performance of the generative networks
#
# 1. Compute the mean squared error between original images and reconstructions
#
# 2. Compute the laplacian variance of the generated images
# 
# 3. Compare similarity between generated images and originals
###############################################################################################################

imgs = np.load('full_vol_norm.npy').astype('float32')
imgs = np.expand_dims(imgs,-1).
xtrain = imgs

imgs_vae = imgs 
imgs_gan = imgs*2-1

idx = np.random.randint(0,imgs.shape[0],100)

## Model

img_dim = (128,128,1)

size = 128

z_dim = 128

epochs = 2000

batch_size = 64

iter_per_epoch = epochs // batch_size 

alpha = 2*10e-9 # Weight of perceptual loss

beta = 1 # Weight of KL-Divergence

# VGG16 model

#pm = keras.models.load_model('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
pm = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(128,128,3)))
#pm.summary()
selected_layers = ['block1_conv1','block2_conv1','block3_conv1']

######### Helper Functions ######

idx = np.random.randint(0,xtrain.shape[0],8*8)

def generate_images(n,encoder,decoder):
	
	fig = plt.figure(figsize=(2*n,n))
	
	originals = xtrain[idx]
	encoded = encoder.predict(originals)
	recons = decoder.predict(encoded)[:,:,:,0]
	originals = originals[:,:,:,0]
	print(originals.shape)
	originals = np.reshape(originals,(8,8,128,128)).transpose(0,2,1,3)
	#originals = np.transpose(originals(0,2,1,3))
	originals = np.reshape(originals,(8*128,8*128))
	recons = np.reshape(recons,(8,8,128,128)).transpose(0,2,1,3)
	#recons = np.transpose(recons)
	recons = np.reshape(recons,(8*128,8*128))

	#images_1 = np.concatenate((originals,recons),axis=1)
	ax = fig.add_subplot(1,2,1)
	ax.imshow(originals)
	ax.set_title('Original Images')

	ax = fig.add_subplot(1,2,2)
	ax.imshow(recons)
	ax.set_title('Reconstructions from Originals')

	plt.savefig('reconstruction.png')


	fig = plt.figure(figsize=(n,n))
	samples = decoder.predict(np.random.standard_normal((n*n,z_dim)))[:,:,:,0]
	samples = np.reshape(samples,(n,n,size,size)).transpose(0,2,1,3)
	samples = np.reshape(samples,(n*size,n*size))

	ax = fig.add_subplot(1,1,1)
	ax.imshow(samples)
	ax.set_title('Generated Images')
	plt.savefig('generation.png')



######### ARCHITECTURE ##########
def sample(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal((K.shape(z_mu)[0], z_dim),)
    return z_mu + K.exp(z_log_sigma) * epsilon


def conv2d_bn_relu(inp,filters,kernel_size,stride,activation=True,bn=True):
	x = Conv2D(filters=filters,kernel_size = kernel_size, strides = stride,padding='same')(inp)
	if(bn == True):
		x = BatchNormalization()(x)
	if(activation==True):
		x = LeakyReLU()(x)

	return x 

class CustomLossLayer(keras.layers.Layer):
	def vae_loss(self, x, z_decoded):
	
		outputs = [pm.get_layer(l).output for l in selected_layers]
		model = Model(pm.input, outputs)


		h1_list = model(x)
		h2_list = model(z_decoded)

		rc_loss = 0
		for h1,h2 in zip(h1_list,h2_list):
			h1 = K.flatten(h1)
			h2 = K.flatten(h2)
			rc_loss = rc_loss + K.sum(K.square(h1-h2),axis=-1)
		#rc_loss = K.mean(rc_loss,axis=-1)
		# KL divergence
		kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

		return K.mean(alpha*rc_loss + beta*kl_loss) 

		# Adds the custom loss
	def call(self, inputs):
		x = inputs[0]
		x_3 = Concatenate(axis=-1)([x,x,x])
		#x_1_chann = x[:,:,:,1]

		z_decoded = inputs[1]
		z_3 = Concatenate(axis=-1)([z_decoded,z_decoded,z_decoded])
		
		
		loss = self.vae_loss(x_3, z_3)
		self.add_loss(loss, inputs=inputs)
		return z_decoded


# Encoder Architecture

x_in = Input(shape=img_dim)
#x_in_loss = Concatenate(axis=-1)([x_in,x_in,x_in])
#print(K.int_shape(x_in_loss))

x = conv2d_bn_relu(x_in,filters=32,kernel_size=4,stride=2)
x = conv2d_bn_relu(x,filters=64,kernel_size=4,stride=2)
x = conv2d_bn_relu(x,filters=128,kernel_size=4,stride=2)
x = conv2d_bn_relu(x,filters=256,kernel_size=4,stride=2)
#x = conv2d_bn_relu(x,filters=512,kernel_size=4,stride=2)

shape_b4_flat = K.int_shape(x)

x_flat = Flatten()(x)

z_mean = Dense(z_dim)(x_flat)
z_log_var = Dense(z_dim)(x_flat)

z = Lambda(sample)([z_mean, z_log_var])

#encoder = Model(x_in,z)

# Decoder Architecture

y_in = Input(shape=K.int_shape(z)[1:])

y = Dense(np.prod(shape_b4_flat[1:]))(y_in)
y = LeakyReLU()(y)

y = Reshape(shape_b4_flat[1:])(y)

y = UpSampling2D()(y)
y = conv2d_bn_relu(y,filters=128,kernel_size=3,stride=1)
y = UpSampling2D()(y)
y = conv2d_bn_relu(y,filters=64,kernel_size=3,stride=1)
y = UpSampling2D()(y)
y = conv2d_bn_relu(y,filters=32,kernel_size=3,stride=1)
y = UpSampling2D()(y)
#y = conv2d_bn_relu(y,filters=1,kernel_size=3,stride=1,activation=False,bn=False)
y = Conv2D(filters=1,kernel_size=3,strides=1,padding='same',activation='sigmoid')(y)

decoder = Model(y_in,y)

recon = decoder(z)
#recon_loss = Concatenate(axis=-1)([recon,recon,recon])
#print(K.int_shape(recon_loss))

end = CustomLossLayer()([x_in, recon])	

vae = Model(x_in,end)


encoder = load_model('../DFC_Vae/models1_22/model12/enc_vgg.h5',custom_objects={CustomLossLayer:CustomLossLayer,z_dim=z_dim})
decoder = load_model('../DFC_Vae/models1_22/model12/dec_vgg.h5')#,custom_objects={CustomLossLayer:CustomLossLayer,z_dim=z_dim})


# 1) MSE Error

# We reconstruct 100 randomly selected images and compute the mean squared error


# Trial



originals = imgs_vae[idx]
encoded = encoder.predict(orignals)
decoded = decoder.predict(encoded)
print(originals.shape)
print(decoded.shape)

mse = K.mean(K.squared(originals-decoded),axis=-1)




def laplacian_variance(images):
	return [cv2.Laplacian(image, cv2.CV_32F).var() for image in images]

################## VAE ###################



################ DFC VAE #################



############### VAE Faces ################



############# DFC VAE Faces ##############



################## GAN ###################



img_dim = (128,128,1)

size = 128

z_dim = 200

epochs = 2000

batch_size = 64

iter_per_epoch = epochs // batch_size 

alpha =2*10e-9 # Weight of perceptual loss

beta = 1 # Weight of KL-Divergence

# VGG16 model

pm = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(128,128,3)))

selected_layers = ['block1_conv1','block2_conv1','block3_conv1']

######### Helper Functions ######
ten = np.random.randint(0,xtrain.shape[0],10000)
idx = np.random.randint(0,xtrain.shape[0],8*8)

def generate_images(n,encoder,decoder,epoch):
	
	fig = plt.figure(figsize=(2*n,n))
	
	originals = xtrain[idx]
	encoded = encoder.predict(originals)
	recons = decoder.predict(encoded)[:,:,:,0]
	originals = originals[:,:,:,0]
	print(originals.shape)
	originals = np.reshape(originals,(8,8,128,128)).transpose(0,2,1,3)
	#originals = np.transpose(originals(0,2,1,3))
	originals = np.reshape(originals,(8*128,8*128))
	recons = np.reshape(recons,(8,8,128,128)).transpose(0,2,1,3)
	#recons = np.transpose(recons)
	recons = np.reshape(recons,(8*128,8*128))

	#images_1 = np.concatenate((originals,recons),axis=1)
	ax = fig.add_subplot(1,2,1)
	ax.imshow(originals*255,cmap='gray')
	ax.set_title('Original Images')

	ax = fig.add_subplot(1,2,2)
	ax.imshow(recons*255,cmap='gray')
	ax.set_title('Reconstructions from Originals')

	plt.savefig('reconstruction_{}'.format(epoch)+'.png')


	fig = plt.figure(figsize=(n,n))
	samples = decoder.predict(np.random.standard_normal((n*n,z_dim)))[:,:,:,0]
	samples = np.reshape(samples,(n,n,size,size)).transpose(0,2,1,3)
	samples = np.reshape(samples,(n*size,n*size))

	ax = fig.add_subplot(1,1,1)
	ax.imshow(samples*255,cmap='gray')
	ax.set_title('Generated Images')
	plt.savefig('generation_{}'.format(epoch)+'.png')



######### ARCHITECTURE ##########
def sample(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal((K.shape(z_mu)[0], z_dim),)
    return z_mu + K.exp(z_log_sigma) * epsilon


def conv2d_bn_relu(inp,filters,kernel_size,stride,activation=True,bn=True):
	x = Conv2D(filters=filters,kernel_size = kernel_size, strides = stride,padding='same')(inp)
	if(bn == True):
		x = BatchNormalization()(x)
	if(activation==True):
		x = LeakyReLU()(x)

	return x 

class CustomLossLayer(keras.layers.Layer):
	def vae_loss(self, x, z_decoded):
	
		outputs = [pm.get_layer(l).output for l in selected_layers]
		model = Model(pm.input, outputs)


		h1_list = model(x)
		h2_list = model(z_decoded)

		rc_loss = 0
		for h1,h2 in zip(h1_list,h2_list):
			h1 = K.flatten(h1)
			h2 = K.flatten(h2)
			rc_loss = rc_loss + K.sum(K.square(h1-h2),axis=-1)
		#rc_loss = K.mean(rc_loss,axis=-1)
		# KL divergence
		kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

		return K.mean(alpha*rc_loss + beta*kl_loss) 

		# Adds the custom loss
	def call(self, inputs):
		x = inputs[0]
		x_3 = Concatenate(axis=-1)([x,x,x])
		#x_1_chann = x[:,:,:,1]

		z_decoded = inputs[1]
		z_3 = Concatenate(axis=-1)([z_decoded,z_decoded,z_decoded])
		
		
		loss = self.vae_loss(x_3, z_3)
		self.add_loss(loss, inputs=inputs)
		return z_decoded


# Encoder Architecture

x_in = Input(shape=img_dim)
#x_in_loss = Concatenate(axis=-1)([x_in,x_in,x_in])
#print(K.int_shape(x_in_loss))

x = conv2d_bn_relu(x_in,filters=32,kernel_size=4,stride=2)
x = conv2d_bn_relu(x,filters=64,kernel_size=4,stride=2)
x = conv2d_bn_relu(x,filters=128,kernel_size=4,stride=2)
x = conv2d_bn_relu(x,filters=256,kernel_size=4,stride=2)
#x = conv2d_bn_relu(x,filters=512,kernel_size=4,stride=2)

shape_b4_flat = K.int_shape(x)

x_flat = Flatten()(x)

z_mean = Dense(z_dim)(x_flat)
z_log_var = Dense(z_dim)(x_flat)

z = Lambda(sample)([z_mean, z_log_var])

#encoder = Model(x_in,z)

# Decoder Architecture

y_in = Input(shape=K.int_shape(z)[1:])

y = Dense(np.prod(shape_b4_flat[1:]))(y_in)
y = LeakyReLU()(y)

y = Reshape(shape_b4_flat[1:])(y)

y = UpSampling2D()(y)
y = conv2d_bn_relu(y,filters=128,kernel_size=3,stride=1)
y = UpSampling2D()(y)
y = conv2d_bn_relu(y,filters=64,kernel_size=3,stride=1)
y = UpSampling2D()(y)
y = conv2d_bn_relu(y,filters=32,kernel_size=3,stride=1)
y = UpSampling2D()(y)
#y = conv2d_bn_relu(y,filters=1,kernel_size=3,stride=1,activation=False,bn=False)
y = Conv2D(filters=1,kernel_size=3,strides=1,padding='same',activation='sigmoid')(y)

decoder = Model(y_in,y)

recon = decoder(z)


end = CustomLossLayer()([x_in, recon])	

vae = Model(x_in,end)

encoder = Model(x_in,z)

hunned = np.random.randint(0,xtrain.shape[0],500)
trials = xtrain[hunned]
noise = np.random.standard_normal((500,z_dim))


row = []
for i in range(10,13):


	direc = 'models/model{}'.format(str(i))
	file = direc+'/'+'info.txt'
	f = open(file,"r")
	line = f.readline()
	line = re.split(r'\t+',line)
	print(line)
	
	name, dimension, beta = line[0], line[1], line[2]
	print(name,dimension,beta)

	#vae.load_weights('models/model{}/vae_vgg.h5'.format(str(i)))
	decoder.load_weights('models/model{}/dec_vgg.h5'.format(str(i)))
	encoder.load_weights('models/model{}/enc_vgg.h5'.format(str(i)))
	images = decoder.predict(noise)[:,:,:,0]
	#gen_imgs = decoder.predict(np.random.standard_normal((10000,z_dim)))*255.
	#fakes = np.concatenate([gen_imgs,gen_imgs,gen_imgs],axis=-1).transpose(0,3,1,2)
	#in_score = get_inception_score(fakes,splits=10)
	#fid_score = get_fid(eval_xtrain[ten],fakes)


	mse = recon_acc()
	div = diversity(images)
	blurr = sum(laplacian_variance(images))/noise.shape[0]
	print(mse,div,blurr)

	row.append([name,dimension,beta[:-1],mse,div,blurr])#,str(in_score),str(fid_score)])
keep_track = pd.DataFrame(row,columns=['Model','Latent Dimension','Beta','MSE','Diversity','Blurriness'])#,'Inception Score','Fid'])
print(keep_track)
keep_track.to_csv('Evaluation',sep='\t')


decoder.load_weights('models/model12/dec_vgg.h5')
encoder.load_weights('models/model12/enc_vgg.h5')

random = np.random.standard_normal((32,z_dim))
fakes = decoder.predict(random)[:,:,:,0]
r12 = np.concatenate(fakes[:8], axis = 1)
r22 = np.concatenate(fakes[8:16], axis = 1)
r32 = np.concatenate(fakes[16:24], axis = 1)
r43 = np.concatenate(fakes[24:], axis = 1)

c1 = np.concatenate([r12, r22, r32, r43], axis = 0)

x = Image.fromarray(np.uint8(c1*255))

x.save("32.jpg")

'''
img = np.zeros((10,128,1280))
for j in range(10):

    one = linear_interpolation(np.expand_dims(xtrain[np.random.randint(0,xtrain.shape[0])],axis=0),np.expand_dims(xtrain[np.random.randint(0,xtrain.shape[0])],axis=0),9)
    img[j] = np.concatenate(one,axis=1)[:,:,0]
print(img.shape)

imgs = np.concatenate(img,axis=0)
print(imgs.shape)

#img = np.concatenate(one,axis=1)[:,:,0]
imsave("latent_space_12.png",imgs)

im1 = imgs[:128,:128]
im2 = imgs[:128,128:256]

print(np.sum((im1-im2)**2)*1/(128.*128.))
'''






