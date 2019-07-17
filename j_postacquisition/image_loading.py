import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from math import log, sqrt, sin
import scipy.signal, scipy.ndimage
import os, sys, time, warnings
import plist_wrapper as jPlist
from image_class import *
from periods import *
from tqdm import *
import tifffile
import hashlib
from PIL import Image, ImageSequence
from pprint import pprint

thisModule = sys.modules[__name__]
thisModule.warnedAboutHittingEdge = False

def SortImagesByPhase(images, key='postacquisition_phase'):
    # Establish an array whose columns represent:
    # - array index
    # - phase
    a = np.zeros((len(images), 2))
    # Note: for the array indexing later on, we need these indices to be int (under python 3, at least...).
    # However the phases in the array 'a' need to be float.
    # We just have to accept that the indices are treated as floats,
    # and cast them to int later on.
    a[:,0] = np.arange(len(images))
    if (key == 'postacquisition_phase'):
        a[:,1] = np.vectorize(images[0].__class__.postacquisition_phase)(images)
    elif (key == 'phase_from_offline_sync_analysis'):
        a[:,1] = np.vectorize(images[0].__class__.phase_from_offline_sync_analysis)(images)
    else:
        assert(key == 'sync_info.phase')
        a[:,1] = np.vectorize(images[0].__class__.rts_phase)(images) % (2*np.pi)

    # Sort by phase-wrapped frame index
    a = a[a[:,1].argsort(kind='mergesort')]     # May as well use a stable sort...
    indices = a[:, 0].astype('int')
    sortedImages = [images[i] for i in a[:, 0].astype('int')]
    return sortedImages


def LoadImageFile(imagePath, loadImageData, downsampleFactor, frameIndex, cropRect, cropFollowsZ, timestampKey):
    # Load an image file, which may contain more than one image - and if a plist is present then it should have
    # the same number of entries

    # First load the plist (which is required to be present)
    plistPath = os.path.splitext(imagePath)[0] + '.plist'
    pl = jPlist.readPlist(plistPath)
    if ((not ('metadata_version' in pl)) or (pl['metadata_version'] == 1)):
        # Metadata version 1 - everything at the root level in the plist
        metadataVersion = 1
        framesMetadata = [pl]
        globalMetadata = [pl]
        numFrames = 1
    else:
        # Metadata version 2 - everything in frames[i]
        # Metadata version 3 - global metadata only appears once
        assert('frames' in pl)
        metadataVersion = pl['metadata_version']
        framesMetadata = pl['frames']
        globalMetadata = pl.copy()
        del globalMetadata['frames']
        numFrames = len(framesMetadata)

    if (loadImageData):
        tif = tifffile.TiffFile(imagePath)
        imageData = tif.asarray()
        if imageData.ndim==2:
            imageData = imageData.reshape([1,imageData.shape[0],imageData.shape[1]])

    # Now iterate over each image, creating the ImageClass objects
    images = []
    for n in range(numFrames):
        if (metadataVersion == 3):
            image = ImageClass_v3()
        else:
            assert(metadataVersion < 3)
            image = ImageClass_v1or2()
        image._path = imagePath
        image._plistPath = plistPath
        image.frameIndex = frameIndex
        frameIndex = frameIndex + 1
        image.metadataVersion = metadataVersion
        image._frameMetadata = framesMetadata[n]
        image._globalMetadata = globalMetadata
        image.timestamp = image.frameKeyPath(timestampKey)

        if (loadImageData):
            # Note that in July 2016 I noticed a major performance issue with img.imread on my mac pro (not on my laptop).
            # I tracked it down to PIL.ImageFile.load, which was calling mmap, and that was taking an abnormally long time.
            # I worked around this issue by editing that code for PIL, and setting use_mmap to False in that function.
            # However, now that is hopefully irrelevant anyway, because I am going to use the 'tiffile' library instead,
            # for reading/writing of multi-page tiffs
            im = imageData[n]
            if (len(im.shape) == 3):    # Colour image
                im = np.sum(im, axis=2)
            # Optional cropping
            # cropRect is ordered (x1, x2, y1, y2)
            if (cropRect is not None):
                cropToUse = cropRect
                if (cropFollowsZ == 'x'):
                    assert(0)   # Not yet implemented
                elif (cropFollowsZ == 'y'):
                    # Read the z coordinate and correct for z motion
                    if ('z_scan' in image._frameMetadata):        # This will need updating if I ever move this to global metadata
                        startZ = image.z_scan()['startZ']
                        z = image.stage_positions()['last_known_z']
                        deltaZ = z - startZ
                    else:
                        deltaZ = 0
                    # For now, I have hard-coded a conversion factor between pixels and z coord
                    # TODO: ideally this would be a parameter that could be adjusted.
                    # However, in practice it is going to remain unchanged unless we mess with the optics.
                    # As a result, I would rather not read the pixel size in the plist, since it's more likely that we won't have
                    # updated that when moving the camera between imaging arms!
                    binning = 2**(image.binning_code())
                    pixelsPerUm = 0.758 / binning
                    correction = int(deltaZ * pixelsPerUm)
                    # Limit correction to full size of image
                    #print(cropToUse[3], correction, im.shape[0])
                    if (cropToUse[3]+correction > im.shape[0]):
                        if (thisModule.warnedAboutHittingEdge == False):
                            print('Hit edge of image during moving-window cropping. Wanted correction', correction, 'for image', imagePath)
                            thisModule.warnedAboutHittingEdge = True
                        correction = im.shape[0] - cropToUse[3]
                    cropToUse = (cropToUse[0], cropToUse[1], cropToUse[2] + correction, cropToUse[3] + correction)
                else:
                    assert(cropFollowsZ is None)

                im = im[cropToUse[2]:cropToUse[3], cropToUse[0]:cropToUse[1]]
            # Option to downsample for speed (but factor will often just be 1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im = scipy.ndimage.interpolation.zoom(im, zoom=1.0/downsampleFactor, order=1)
        else:
            im = None
        image.image = im
        images.append(image)

    return images

def MyHash(runningHash, b):
    # Note that in python 3 apparently hash() is for some reason not unique across invocations of python (ouch!)
    # We should be ok if we use hashlib instead
    if isinstance(b, str) == False:
        b = str(b)
    runningHash.update(b.encode('utf-8'))

def LoadOrCacheImagesFromFileList(fileList, loadImageData, downsampleFactor, frameIndexStart, periodRange, plotAllPeriods, cropRect, cropFollowsZ, timestampKey, saveCache=True, lateStart=0, earlyTruncation = -1, log=True):
    # Loads a sequence of images from fileList
    # For performance reasons, we cache the numpy array so we can load faster if rerunning the same code

    if (len(fileList) == 0):
        return (None, np.nan)

    # filelist is expected to be sorted, and on Linux this will not be the case by default
    # fileList = sorted(fileList)  # This won't correctly sort, e.g., 99.tif and 100.tif
    fileList = sorted(fileList,key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    # Determine the cache file name appropriate for this fileList
    hashObject = hashlib.md5()
    for f in fileList:
        MyHash(hashObject, f)
    MyHash(hashObject, downsampleFactor)
    MyHash(hashObject, frameIndexStart)
    MyHash(hashObject, cropRect)
    MyHash(hashObject, cropFollowsZ)
    MyHash(hashObject, timestampKey)
    MyHash(hashObject, loadImageData)
    MyHash(hashObject, 'v3')#differentiates between hashes on newer ImageClass format (2018-04-11)
    if lateStart > 0 or earlyTruncation > 0:
        MyHash(hashObject, lateStart)
        MyHash(hashObject, earlyTruncation)
    hashResult = hashObject.hexdigest()[-16:]
    # Note that EditOriginalPlists makes assumptions about the form of this cache file name
    cacheName = "%s/%s_%s.npy" % (os.path.dirname(fileList[0]), os.path.basename((os.path.split(fileList[0]))[0]), hashResult)

    try:
        #Try loading an .npy file of the image information.
        if log:
            print("Loading from cache: %s." % cacheName)
        images = np.load(cacheName)
    except IOError:
        if log:
            print("  Missing npy file: %s. Loading files." % cacheName)
        images = []
        counter = 0
        for imagePath in tqdm(fileList, desc='loading images'):
            queued = len(LoadImageFile(imagePath, False, downsampleFactor, frameIndexStart + counter, cropRect=cropRect, cropFollowsZ=cropFollowsZ, timestampKey=timestampKey))
            remainder = lateStart - queued - counter
            if remainder < 0:
                images.extend(LoadImageFile(imagePath, loadImageData, downsampleFactor, frameIndexStart + counter, cropRect=cropRect, cropFollowsZ=cropFollowsZ, timestampKey=timestampKey))
            if lateStart > 0 and remainder>0 and remainder < queued:
                if log:
                    print('Late start requested.')
                images = images[remainder:]
            counter = counter + queued
            if earlyTruncation > 0 and len(images) > (earlyTruncation - lateStart):
                if log:
                    print('Early truncation requested.')
                images = images[:(earlyTruncation - lateStart)]
                break
        images = np.array(images)
        if saveCache:
            try:
                np.save(cacheName, images)
            except IOError:
                print("Harmless warning: unable to save cache (probably a read-only volume?)")

    if (loadImageData and (periodRange is not None)):
        # Estimate approximate period.
        numImagesToUseForPeriodEstimation = min(int(periodRange[-1] * 12), len(images))
        averagePeriod = EstablishPeriodForImageSequence(images[0:numImagesToUseForPeriodEstimation], periodRange = periodRange, plotAllPeriods = plotAllPeriods)
        if log:
            print('estimated average(ish) period', averagePeriod, 'from first', numImagesToUseForPeriodEstimation, 'images')
    else:
        averagePeriod = np.nan

    return (images, averagePeriod)


def LoadAllImages(path, loadImageData = True, downsampleFactor = 1, frameIndexStart = 0, earlyTruncation = -1, periodRange = None, plotAllPeriods=False, cropRect=None, cropFollowsZ=None, timestampKey='time_processing_started', lateStart=0, saveCache=True, log=True):
    # Load all images found in the directory at 'path'.
    # We also do a rough estimate the average heart period at the start of the dataset (because that is a useful guide for later sync processing)
    # We skip any invisible files (starting with '.') since on OS X the Preview program sometimes creates invisible mirror files when opening tiff stacks.
    fileList = []
    for file in os.listdir(path):
        if ((not file.startswith(".")) and (file.endswith(".tif") or file.endswith(".tiff"))):
            fileList.append(path+'/'+file)
    return LoadOrCacheImagesFromFileList(fileList, loadImageData, downsampleFactor, frameIndexStart, periodRange, plotAllPeriods, cropRect, cropFollowsZ, timestampKey,saveCache,lateStart=lateStart,earlyTruncation=earlyTruncation,log=log)


def LoadImages(path, format, firstImage, numImagesToProcess, loadImageData = True, downsampleFactor = 1, frameIndexStart = 0, periodRange = np.arange(20, 50, 0.1), plotAllPeriods=False, cropRect=None, cropFollowsZ=None, timestampKey='time_processing_started',saveCache=True, log=True):
    # Load the source images and metadata for all images found at 'path' that match the supplied filename format with indices given by firstImage and numImagesToProcess.
    # We also do a rough estimate the average heart period at the start of the dataset (because that is a useful guide for later sync processing)
    fileList = []
    for i in range(firstImage, firstImage+numImagesToProcess):
        imagePath = (format % (path, i))
        fileList.append(imagePath)
    return LoadOrCacheImagesFromFileList(fileList, loadImageData, downsampleFactor, frameIndexStart, periodRange, plotAllPeriods, cropRect, cropFollowsZ, timestampKey,saveCache,log=log)

def ImagesToArray(images):
    # Converts a list of image objects into a 3D numpy array
    npIm = None
    for im in images:
        if npIm is None:
            npIm = im.image[np.newaxis,:,:]
        else:
            npIm = np.append(npIm, im.image[np.newaxis,:,:], axis=0)
    return npIm


#
# JMT 1.3.18: I am not sure who uses this next function, and for what, but it will need updating to account for the new plist format
# I am happy to do that, but since it's probably useful to use my C code to process the zyla spool files anyway,
# it may be easier to achieve this functionality during that same processing, rather than in a separate function here
#
#def StripToMinimalMetadataForZyla(path, keepKeys=['frame_number', 'time_exposed', 'time_received', 'timestamp', 'z_scan', 'phase_from_offline_sync_analysis', 'postacquisition_phase', 'experimenter_name', 'specimen_description']):
#    (images, dummy) = LoadAllImages(path, loadImageData=False, periodRange=None)
#    DeleteAnyCachesInDir(path)
#    for im in tqdm(images, 'Stripping plists'):
#        newPlist = dict()
#        for key in keepKeys:
#            if key in im.plist:
#                newPlist[key] = im.plist[key]
#        im.plist = newPlist
#        jPlist.writePlist(newPlist, im.plistPath)
