from PIL import Image 
import numpy as np 
import scipy
from scipy.misc import imsave
import importDicoms as iD 
from scipy.stats import mode
import os



def dicom_to_array(path,single_images=True,name='labels.npy'):
	images = iD.loadDicomFromFolderIn3D(path)
	if(single_images):
		dim = images.shape
		array = np.zeros((dim[0]*dim[3],dim[1],dim[2]))

		for i in range(dim[0]):
			for j in range(dim[3]):
				array[i*40+j,:,:]=images[i,:,:,j]

		np.save(name,array)
	if(single_images):
		return array	
	return images


def crop_z(array,lower_bound,upper_bound,imgs_per_group,name):

	dim = array.shape
	patients = dim[0]//imgs_per_group
	new_imgs_per_group = upper_bound-lower_bound + 1
	cropped = np.zeros((new_imgs_per_group*patients,dim[1],dim[2]))
	for i in range(patients):
		for j in range(new_imgs_per_group):
			cropped[i*new_imgs_per_group+j,:,:]=array[i*imgs_per_group+j+lower_bound,:,:]

	np.save(name,cropped)

	return cropped

def get_labels(array):
	dim = array.shape
	labels = np.zeros((dim[0],1))

	for i in range(dim[0]):
		if(np.amax(array[i,:,:])>0):
			labels[i]=1

	np.save('labels.npy',labels)
	return labels

def crop_labels(array,lower_bound, upper_bound, imgs_per_group, name):
	dim = array.shape
	patients = dim[0]//imgs_per_group
	new_imgs_per_group = upper_bound-lower_bound + 1
	cropped = np.zeros((new_imgs_per_group*patients,1))
	for i in range(patients):
		for j in range(new_imgs_per_group):
			cropped[i*new_imgs_per_group+j]=array[i*imgs_per_group+j+lower_bound]

	np.save(name,cropped)

	return cropped

def image_to_patient(array,images_per_patient):
	dim = array.shape 
	n_patients = dim[0]//images_per_patient
	#enlarged = np.zeros((n_patients,128,128*images_per_patient))
	#for i in range(n_patients):
	#	for j in range(images_per_patient):
	enlarged = array.reshape((n_patients,images_per_patient,dim[1],dim[2])).transpose(0,2,1,3).reshape((n_patients,dim[1],dim[2]*images_per_patient))#.transpose(2,1,0)

	return enlarged

def crop(array):
	dim = array.shape 
	return True

def volume_normalization(array):
	dim = array.shape
	volume_size = 26
	number_volumes = dim[0]//26
	normalize = np.zeros(dim)
	for j in range(number_volumes):
		vol = array[j*26:(j+1)*26]
		vol = np.clip(vol,0,np.percentile(vol,99))
		vol = vol/np.amax(vol)
		normalize[j*26:(j+1)*26]=vol
	#normalize = normalize/np.amax(normalize)

	return normalize

def histogram_norm(array):
	underThreshold = 40
	strokeThreshold = 200
	upperThreshold = 500
	dim = array.shape
	number_vols = dim[0]//26
	X_hist = np.zeros(dim)
	for i in range(number_vols):
		print(i)
		X = array[i*26:(i+1)*26]
		oldX = X
		mask = X > underThreshold
		mask.astype(np.int)
		X = np.multiply(X.clip(0,upperThreshold),mask)

		mask_aboveStrokeThreshold = X>strokeThreshold 

		X_left_hemisphere = X[:,0:64,:]
		X_right_hemisphere = X[:,64:128,:]

		leftCount = mask_aboveStrokeThreshold[:,0:64,:].sum()
		rightCount = mask_aboveStrokeThreshold[:,64:128,:].sum()

		if(leftCount>=rightCount):
			normalizationMean = np.mean(X_right_hemisphere[np.nonzero(X_right_hemisphere)])
		else:
			normalizationMean = np.mean(X_left_hemisphere[np.nonzero(X_left_hemisphere)])

		X = np.divide(X,normalizationMean)

		X_rescaled = np.divide(np.subtract(X,X.min()),np.subtract(X.max(),X.min()))
		X_rescaled = np.subtract(X_rescaled, mode(X_rescaled[np.nonzero(X_rescaled)],axis=None)[0][0])

		mask = np.multiply(mask,100)

		X_rescaled = np.multiply(X_rescaled,mask)

		X_hist[i*26:(i+1)*26]=X_rescaled

	np.save('imgs_hist_norm2.npy',X_hist)

	return X_hist

def full_vol_norm(array):
	vol = array
	vol = np.clip(vol,0,np.percentile(vol,95))
	vol = vol/np.amax(vol)

	return vol

def images_as_npy(array,directory_name):
	dim = array.shape
	if not os.path.exists(directory_name):
		os.mkdir(directory_name)
	for i in range(dim[0]):
		print(str(i)+'/'+str(dim[0]))
		img = np.expand_dims(array[i],axis=-1)
		img = np.expand_dims(img,axis=0)
		np.save(directory_name+'/image{}.npy'.format(i),img)

	return True

#arr = np.load('strokes_pos.npy')
#images_as_npy(arr[:3400],'train_strokes')

#print('Starting with images:')
#strokes = dicom_to_array(path,'strokes_ims.npy')
#print('Loaded, cropping now:')
#strokes = crop_z(strokes,7,32,40,'strokes_cropped.npy')
#print('Cropped, normalizing now:')
#strokes = volume_normalization(strokes)
#np.save('strokes.npy',strokes)

#arr = np.load('strokes_cropped.npy')
#c=histogram_norm(arr)
#lab = np.load('stroke_labels.npy')
#a = np.where(lab==1)

#arr = arr[a[0]]
#arr = volume_normalization(arr)
#np.save('strokes_pos_4.npy',arr)
#np.save('strokes_normalized.npy',arr)


#print('Starting with ground thruths:')
#labels = dicom_to_array(path2,'stroke_gts.npy')
#labels = np.load('stroke_gts.npy')
#print('Computing labels:')
#labels = get_labels(labels)
#print('Loaded, cropping now:')
#labels = crop_labels(labels,7,32,40,'stroke_cropped_gts.npy')
#np.save('stroke_labels.npy',labels)


#lab  = np.load('stroke_labels.npy')
#print(lab.shape)
#a = np.where(lab==1)
#print(a[0])

#strokes = np.load('strokes.npy')
#print(strokes.shape)
#with_stroke = strokes[a[0]]
#print(with_stroke.shape)
#arr = np.load('positive_strokes.npy')
#print(arr.shape)



c = np.load('../../GoodData/normal_brians.npy').astype('float32')
#c = np.load('brains_5_32.npy').astype('float32')
c = np.expand_dims(c,axis=-1)


layer1 = c[9*26:10*26]
layer2 = c[255*26:256*26]
layer3 = c[1000*26:1001*26]
layer4 = c[700*26:701*26]
layer5 = c[95*26:96*26]

layer1 = np.reshape(layer1,(1,26,128,128)).transpose(0,2,1,3).reshape(128,26*128)
layer2 = np.reshape(layer2,(1,26,128,128)).transpose(0,2,1,3).reshape(128,26*128)
layer3 = np.reshape(layer3,(1,26,128,128)).transpose(0,2,1,3).reshape(128,26*128)
layer4 = np.reshape(layer4,(1,26,128,128)).transpose(0,2,1,3).reshape(128,26*128)
layer5 = np.reshape(layer5,(1,26,128,128)).transpose(0,2,1,3).reshape(128,26*128)

layers = np.concatenate((layer1,layer2,layer3,layer4,layer5),axis=0)
#imsave('full_lat.png',layers)
x = Image.fromarray(np.uint8(layers*255))
x.save("dataset_example2.png")









