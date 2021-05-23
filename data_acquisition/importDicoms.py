import pydicom
# import keras.backend as K

from keras import backend as K

import glob
import numpy as np
from PIL import Image

#
# pathToDicomFolpathToDicomFolderder = '/Users/christian/Desktop/dataToLoad/ims/0';
#
# listOfPathToCases
# listOf
#
# listOfPathToDicoms = sorted(glob.glob(pathToDicomFolder + '/*'))
#
# caseNames = set();
# for i in range(len(listOfPathToDicoms)):
#     caseNames.add(listOfPathToDicoms[i][-16:-8])
#
# listOfCases = sorted(list(caseNames))
#



def listOfCasesInDicomFolder(pathToDicomFolder):

    # listOfPathToDicoms = sorted(glob.glob(pathToDicomFolder + '/*'))
    listOfPathToDicoms = glob.glob(pathToDicomFolder + '/*')

    #use the property that set have unique elements
    caseNames = set()
    for i in range(len(listOfPathToDicoms)):
        caseNames.add(listOfPathToDicoms[i][-16:-8])

    return sorted(list(caseNames))





def listOfSlicesNumbersInDicomFolder(pathToDicomFolder):

    # listOfPathToDicoms = sorted(glob.glob(pathToDicomFolder + '/*'))
    listOfPathToDicoms = glob.glob(pathToDicomFolder + '/*')

    #use the property that set have unique elements
    caseNumbers = set()
    for i in range(len(listOfPathToDicoms)):
        caseNumbers.add(listOfPathToDicoms[i][-7:-4])

    return sorted(list(caseNumbers))




def listOfSlicesNumbersSingleCaseInDicomFolder(pathToDicomFolder,caseNumber):

    # returns the list of slice numbers for a given caseNumber in the folder pathToDicomFolder
    # the files in the Dicom folder must be saved in the form 00100001_001.dcm
    # with 00100001 the caseNumber and 001 the sliceNumber

    listOfPathToDicoms = glob.glob(pathToDicomFolder + '/' + caseNumber + '*')

    #use the property that set have unique elements
    caseNumbers = set()
    for i in range(len(listOfPathToDicoms)):
        caseNumbers.add(listOfPathToDicoms[i][-7:-4])

    return sorted(list(caseNumbers))




def loadDicomFromFolderIn3D(pathToDicomFolder):

# this function loads the dicoms from a Folder and save it in a 4D array: caseNumber, x, y, z
# the files in the Dicom folder must be saved in the form 00100001_001.dcm
# with 00100001 the caseNumber and 001 the sliceNumber

    _listOfCasesInDicomFolder = listOfCasesInDicomFolder(pathToDicomFolder)
    _listOfSlicesNumbersInDicomFolder = listOfSlicesNumbersInDicomFolder(pathToDicomFolder)


    #get the first dicomImage and read the size in x and y direction
    pathToFirstDicom = pathToDicomFolder + '/' + _listOfCasesInDicomFolder[1] + '_' + _listOfSlicesNumbersInDicomFolder[1] + '.dcm'
    ds_firstDicom = pydicom.dcmread(pathToFirstDicom)
    pix = ds_firstDicom.pixel_array

    #initialize image_array to the right size
    image_array = np.zeros((len(_listOfCasesInDicomFolder), pix.shape[0], pix.shape[1], len(_listOfSlicesNumbersInDicomFolder)))


    for i in range(len(_listOfCasesInDicomFolder)):
        print('load ' + _listOfCasesInDicomFolder[i])

        for j in range(len(_listOfSlicesNumbersInDicomFolder)):

            pathToDicom = pathToDicomFolder + '/' + _listOfCasesInDicomFolder[i] + '_' + _listOfSlicesNumbersInDicomFolder[j] + '.dcm'
            # print(pathToDicom)

            ds = pydicom.dcmread(pathToDicom)
            pix = ds.pixel_array
            image_array[i, :, :, j] = pix


    return image_array





def loadDicomFromFolder(pathToDicomFolder):

    #this function loads the dicoms in pathToDicomFolder as 3D image numpy array
    listOfPathToDicoms = sorted(glob.glob(pathToDicomFolder + '/*'))

# This is to make the function more general, but it is a bit slower than define sizeOfImage only once
    ds_firstDicom = pydicom.dcmread(listOfPathToDicoms[1])
    pix = ds_firstDicom.pixel_array
    image_array = np.zeros((len(listOfPathToDicoms), pix.shape[0], pix.shape[1]))

    # image_array = np.zeros((len(listOfPathToDicoms),sizeOfImage[0],sizeOfImage[1]))

    for i in range(len(listOfPathToDicoms)):
        print('load ' + listOfPathToDicoms[i])
        ds = pydicom.dcmread(listOfPathToDicoms[i])
        pix = ds.pixel_array
        image_array[i,:,:] = pix

    return image_array





def img_to_array(img, dim_ordering=K.image_dim_ordering()):
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


def general_image_reader(filepath, target_mode=None, target_size=None, dim_ordering=K.image_dim_ordering(), **kwargs):
    if filepath[-3:] == 'dcm':
        ds = pydicom.dcmread(filepath)
        img = ds.pixel_array
    else:
        # img = load_img(filepath, target_mode=target_mode, target_size=target_size)
        img = Image.open(filepath)
        # img = cropAndNormalize(img)
    return img_to_array(img, dim_ordering=dim_ordering)



def listOfSlicesNumbers(pathToDicomFolder, caseNumber):
    # returns the list of slice numbers for a given caseNumber in the folder pathToDicomFolder
    # the files in the Dicom folder must be saved in the form 00100001_001.dcm
    # with 00100001 the caseNumber and 001 the sliceNumber

    listOfPathToDicoms = glob.glob(pathToDicomFolder + '/' + caseNumber + '*')

    # use the property that set have unique elements
    caseNumbers = set()
    for i in range(len(listOfPathToDicoms)):
        caseNumbers.add(listOfPathToDicoms[i][-7:-4])

    return sorted(list(caseNumbers))


def load3DImages(pathToDicomFolder,caseNumber, imageType = '.dcm'):
    # this function loads the dicoms from a Folder and save it in a 3D array: x, y, z
    # the files in the Dicom array must be saved in the form 00100001_001.dcm
    # with 00100001 the caseNumber and 001 the sliceNumber

    _listOfSlicesNumbers = listOfSlicesNumbers(pathToDicomFolder, caseNumber)

    # get the first dicomImage and read the size in x and y direction
    pathToFirstDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[1] + imageType
    ds_firstDicom = pydicom.dcmread(pathToFirstDicom)
    pix = ds_firstDicom.pixel_array

    # initialize image_array to the right size
    image_array = np.zeros((pix.shape[0], pix.shape[1], len(_listOfSlicesNumbers)))

    for j in range(len(_listOfSlicesNumbers)):

        # pathToDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + self.imageType
        # ds = pydicom.dcmread(pathToDicom)
        # pix = ds.pixel_array
        # image_array[:, :, j] = pix

        pathToImage = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + imageType

        if imageType == '.dcm':
            ds = pydicom.dcmread(pathToImage)
            pix = ds.pixel_array
            image_array[:, :, j] = pix

        else:
            image_array[:, :, j] = Image.open(pathToImage)

    return image_array



def load3DImagesAndCrop(pathToDicomFolder, caseNumber, imageType='.dcm'):
        # this function loads the dicoms from a Folder and save it in a 3D array: x, y, z
        # the files in the Dicom array must be saved in the form 00100001_001.dcm
        # with 00100001 the caseNumber and 001 the sliceNumber

        _listOfSlicesNumbers = listOfSlicesNumbers(pathToDicomFolder, caseNumber)

        # get the first dicomImage and read the size in x and y direction
        pathToFirstDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[1] + imageType
        ds_firstDicom = pydicom.dcmread(pathToFirstDicom)
        pix = ds_firstDicom.pixel_array

        # initialize image_array to the right size
        image_array = np.zeros((pix.shape[0], pix.shape[1], len(_listOfSlicesNumbers)))

        for j in range(len(_listOfSlicesNumbers)):

            # pathToDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + self.imageType
            # ds = pydicom.dcmread(pathToDicom)
            # pix = ds.pixel_array
            # image_array[:, :, j] = pix

            pathToImage = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + imageType

            if imageType == '.dcm':
                ds = pydicom.dcmread(pathToImage)
                pix = ds.pixel_array
                image_array[:, :, j] = pix

            else:
                image_array[:, :, j] = Image.open(pathToImage)


        # we crop down to 96 (26:122) x 80 (25:105)
        #  this permit a network depth of 3

        return image_array[26:122, 25:105, :]




