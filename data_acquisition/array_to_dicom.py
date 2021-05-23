def writeSyntheticDicomImages(baseDirectory,synthDirectory, imageArray, lesionNr, edemaAmount, reconMethod, frameNumber = 0, ):

	""" Generate a folder with dicom-slices from a given image-array (3D)
	based on a real dicom dataset

	Parameters:
	-----------
	- baseDirectory: base directory of case
	- synthDirectory: base directory of synthetic cases
	- imageArray: 3D-image array of dimension (width, length, height), to be written
	- lesionNr: number of lesion which was the base for the simulation
	- edemaAmount: amount of edema in resulting image

	Returns:
	--------
	- None 
	"""

	# Create folder for synthetic data

	folder_synth_base = synthDirectory + 'DWI_ACPC_Edema/' 

	createFolderIfNotExisting(folder_synth_base)

	folder_synth = folder_synth_base + str(lesionNr) +'_' + str(edemaAmount) + '/' + str(frameNumber) + '/'

	createFolderIfNotExisting(folder_synth)

	# Retrieve number of levels and filenames

	base = baseDirectory + parenchymaDir

	srcDir = os.fsencode(base)

	fileList = os.listdir(srcDir)


	# Iterate over slices (0-39)

	for count, file in enumerate(sorted(fileList)) :

		# File Handling

		filename = os.fsdecode(file)

		file_orig = baseDirectory + parenchymaDir + filename 

		file_synth = folder_synth + filename 


		# Load original dataset => reuse dicom meta data

		dataset_orig = pd.dcmread(file_orig)



		# Retrieve image slice for according level

		pixelData = imageArray[:,:,count]


		# Generate dicom-dataset for synthetic image -> adapt header data

		dataset_synth = dataset_orig

		dataset_synth.LargestImagePixelValue = np.max(pixelData).astype(int)

		dataset_synth.SeriesDescription = dataset_orig.SeriesDescription[:-1] +	"_" + str(frameNumber).zfill(2)

		dataset_synth.SeriesInstanceUID = dataset_orig.SeriesInstanceUID + '.' + str(lesionNr) + '.' + str(edemaAmount) + '.' + str(frameNumber)

		#dataset_synth.SOPInstanceUID = dataset_orig.SOPInstanceUID + '.' + str(lesionNr) + '.' + str(edemaAmount) + '.' + str(frameNumber)

		dataset_synth.SOPInstanceUID = dataset_orig.SOPInstanceUID + '.' + str(lesionNr) + '.' + str(edemaAmount)

		#dataset_synth.PatientName = str(dataset_orig.PatientName) + ' - SYN (L: ' + str(lesionNr) + ', E: ' + str(edemaAmount) + ', F: '+ str(frameNumber).zfill(2) +')'

		dataset_synth.PatientName = str(dataset_orig.PatientName) + ' - SYN (L: ' +str(lesionNr).zfill(3) +', E: ' + ', M: '+ reconMethod +')'

		dataset_synth.PixelData = pixelData.astype(np.uint16).tobytes()

		dataset_synth.Rows, dataset_synth.Columns = pixelData.shape

		dataset_synth.save_as(file_synth)


	return True