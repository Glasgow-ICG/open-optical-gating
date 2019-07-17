import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.signal, scipy.ndimage
import plist_wrapper as jPlist
import os, errno, sys
from PIL import Image
from image_class import *
import image_loading as jil
from tqdm import *
import tifffile
sys.path.insert(0, '../cjn-python-emulator/')
import helper as hlp

def SaveImageBatchToFile(imageBatch, imageBatchCounter, destFolder, globalMetadata, vmax, appendOriginalFrameNum):
    # Determine file type and filename
    if (vmax <= 255):
        typeString = 'uint8'
    else:
        typeString = 'uint16'
    if appendOriginalFrameNum:
        # Alter the filename to refer back to the frame number of the original file
        # (specialist feature that is occasionally useful for cross-referencing,
        #  but would not normally be used)
        stem = '%06d_%06d' % (imageBatchCounter, im.frameKeyPath('frame_number'))
    else:
        stem = '%06d' % imageBatchCounter

    # Save the images as a multi-page tiff
    tifffile.imsave(destFolder+('/%s.tif' % stem), hlp.convertObj(imageBatch)[0].astype(typeString))

    if globalMetadata is not None:
        # Save the metadata
        plistPath = destFolder+('/%s.plist' % stem)
        framesMetadata = []
        for im in imageBatch:
            framesMetadata.append(im._frameMetadata)
        metadataVersionToUse = imageBatch[0].metadataVersion
        if metadataVersionToUse == 1:
            metadataVersionToUse = 2        # We resave version 1 as version 2 (to permit batching of multiple frames into one file)
        SaveMetadata(framesMetadata, globalMetadata, metadataVersionToUse, plistPath)

def SaveImagesToFolder(images, destFolder, withMetadata=False, vmax=255, appendOriginalFrameNum=False, maxFileSize=300e6, maxImagesPerFile=1000):
    # To save the plists, we override the original _plistPath property (if there is one),
    # but restore it after we have finished.
    os.system('mkdir -p "%s"' % destFolder)
    imageBatch = []
    imageBatchCounter = 0
    bytesPerPixel = 2 if vmax > 255 else 1
    for counter, im in enumerate(images):
        # First decide if we have a complete file ready to save.
        # We do not if we are under maxImagesPerFile and we have room for another image before hitting maxFileSize
        if (len(imageBatch) > 0):
            projectedFileSize = imageBatch[0].image.size * bytesPerPixel * (len(imageBatch)+1)
            globalMetadataMismatch = False
            thisGlobalMetadata = im._globalMetadata
            if counter > 0:
                prevGlobalMetadata = images[counter-1]._globalMetadata
            if (im.metadataVersion == 1) or (withMetadata == False):
                thisGlobalMetadata = None
            else:
                if (counter > 0) and (thisGlobalMetadata != prevGlobalMetadata):
                    globalMetadataMismatch = True
            if (len(imageBatch) >= maxImagesPerFile) or (projectedFileSize > maxFileSize) or globalMetadataMismatch:
                SaveImageBatchToFile(imageBatch, imageBatchCounter, destFolder, thisGlobalMetadata, vmax, appendOriginalFrameNum)
                imageBatch = []
                imageBatchCounter = counter
        imageBatch.append(im)

    # Save any images not yet saved
    SaveImageBatchToFile(imageBatch, imageBatchCounter, destFolder, imageBatch[0]._globalMetadata, vmax, appendOriginalFrameNum)

    # Reset the plist paths to what they were originally
    for im in images:
        if hasattr(im, '_oldPlistPath'):
            im._plistPath = im._oldPlistPath
            del im._oldPlistPath

def DeleteAnyCachesInDir(theDir):
    for file in os.listdir(theDir):
        if (file.endswith(".npy")):
            print('deleting stale cache file %s' % file)
            os.remove('%s/%s' % (theDir, file))

def SaveMetadata(framesMetadata, globalMetadata, imageMetadataVersion, plistPath):
    # Assemble the metadata provided in the input parameters, and save at plistPath
    plist = dict()
    plist['frames'] = framesMetadata
    if (imageMetadataVersion == 1):
        # This was loaded as metadata version 1, but we are actually re-saving as version 2,
        # because we are using the 'frames' array
        plist['metadata_version'] = 2
    else:
        plist['metadata_version'] = imageMetadataVersion
        if (imageMetadataVersion >= 3):
            plist.update(globalMetadata)    # Adds globalMetadata dictionary to the dictionary 'plist'
    jPlist.writePlist(plist, plistPath)

def ResaveMetadataAfterEditing(images):
    # Modify existing .plist files on disk associated with the 'images' array.
    # This function is called after the calling code has modified the metadata dictionaries associated with the 'images' array.
    # It will maintain whatever file layout was there when the data was earlier loaded from disk.
    prevPlistPath = None
    framesMetadata = []
    for n in tqdm(range(len(images)), desc='save plists'):
        im = images[n]
        assert(im._plistPath is not None)
        if ((prevPlistPath is not None) and (im._plistPath != prevPlistPath)):
            SaveMetadata(framesMetadata, prevGlobalMetadata, prevMetadataVersion, prevPlistPath)
            framesMetadata = []
        prevPlistPath = im._plistPath
        prevGlobalMetadata = im._globalMetadata
        prevMetadataVersion = im.metadataVersion
        framesMetadata.append(im._frameMetadata)
    if (prevPlistPath is not None):
        SaveMetadata(framesMetadata, prevGlobalMetadata, prevMetadataVersion, prevPlistPath)

def GenerateFakePlists(basePath, framerate=1000.0, stemLength=0, fakeZScan=False):
    # This function generates plist files to accompany tiff files that do not actually have real metadata
    # associated with them (e.g. files that were not originally recorded via Spim GUI).
    # We do not generate full metadata, not by a long way - we just generate/invent a few important keys
    # that other bits of code expect to be present.
    counter = 0
    for file in os.listdir(basePath):
        if (file.endswith(".tif") or file.endswith(".tiff")):
            imagePath = basePath+'/'+file
            plistPath = os.path.splitext(imagePath)[0] + '.plist'
            pl = dict()
            try:
                # Attempt to discern a frame number from the filename
                frameNumber = int(os.path.splitext(os.path.basename(imagePath))[0][stemLength:])
            except:
                # ... but failing that just make one up
                frameNumber = counter
            # Use frame number to synthesize fake timestamps (based on the framerate provided by the caller)
            pl['time_processing_started'] = frameNumber / float(framerate)
            pl['time_received'] = frameNumber / float(framerate)
            if fakeZScan:
                # Optional hacky workaround to provide some dummy information to satisfy StackViewer
                pl['z_scan'] = dict()
                pl['z_scan']['speed1'] = 0
                # Include information about stage positions, as if it was a smooth z scan
                pl['stage_positions'] = dict()
                pl['stage_positions']['last_known_z'] = counter * 1.0
                pl['stage_positions']['last_known_z_time'] = pl['time_processing_started']
            counter = counter + 1
            
            jPlist.writePlist(pl, plistPath)

def mkdir_p(path):
   # Ensure directory at 'path' exists. It may already exist.
   # Code from stackoverflow
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def PickOutPhasesFromLieblingStack(basePath, fluorFolders, destRoot, numPhases, phaseKey='phase_from_offline_sync_analysis', timestampKey='time_exposed'):
    # Trawl through a phase-stamped Liebling-style stack and build up 'numPhases' different stacks,
    # each using the closest-matching frame (in terms of phase) from the overall dataset
    for fluorFolder in fluorFolders:
        # Load the metadata for the entire stack (*not* the images themselves, which may not all fit in memory)
        (allImages, dummy) = jil.LoadAllImages('%s/%s' % (basePath, fluorFolder), loadImageData = False)
        # Iterate through identifying each z plane that we will use
        zPlanes = []
        sourceFilesForZ = []
        for im in allImages:
            if im.z_scan()['stepwise_definitely_stationary']:
                thisZ = im.stage_positions()['last_known_z']
                if (len(zPlanes) == 0) or thisZ != zPlanes[-1]:
                    zPlanes.append(thisZ)
                    sourceFilesForZ.append([im._path])
                else:
                    if (sourceFilesForZ[-1][-1] != im._path):
                        sourceFilesForZ[-1].append(im._path)
                            
        delta = 2.0*np.pi / numPhases
        targetPhases = np.arange(delta/2, 2.0*np.pi, delta)
                        
        # Now sort through picking out frames for our synchronized stack.
        # Design choice: it makes life a lot easier if we save an entire stack (one phase) in one go,
        # but it would also be very inefficient if we loaded batches of files by phase rather than by z.
        # Because I want both these things, I build up the *entire* output stack in memory before saving it.
        # Hopefully we won't run out of memory...!

        # Create an empty list of results (working around the fact that ([None]*len(zPlanes))*targetPhases.shape[0] duplicates the list object, which is not what I want!)
        result = []
        for i in range(targetPhases.shape[0]):
            result.append([None]*len(zPlanes))   # appending a new list!

        # Process source frames in batches by z
        for zi in range(len(zPlanes)):
            z = zPlanes[zi]
            print('Processing z=%.3lf'%z)
            sourceFiles = sourceFilesForZ[zi]
            # Load the entire contents of the relevant source files (which may be multi-page tiffs, but that's all we can easily do...)
            # and then sort and strip out any we don't want
            (images, _) = jil.LoadOrCacheImagesFromFileList(sourceFiles, True, 1, 0, None, False, None, False, timestampKey, saveCache=False)
            for j in reversed(range(len(images))):
                try:
                    invalidPhase = InvalidPhase(images[j].frameKeyPath(phaseKey))
                except:
                    invalidPhase = True
                if ((images[j].stage_positions()['last_known_z'] != z) or (images[j].z_scan()['stepwise_definitely_stationary'] == False) or invalidPhase):
                    images = np.delete(images, j)
            if (phaseKey == 'postacquisition_phase'):
                phasesForImages = np.vectorize(images[0].__class__.postacquisition_phase)(images)
            else:
                assert(phaseKey == 'phase_from_offline_sync_analysis')
                phasesForImages = np.vectorize(images[0].__class__.phase_from_offline_sync_analysis)(images)
            for p in range(len(targetPhases)):
                # Search through remaining images for the closest match by phase.
                # Note that I don't handle the wraparound between 0 and 2pi here,
                # which should pretty much be fine as long as phases represent the middle of phase bins.
                diffs = np.abs(phasesForImages-targetPhases[p])
                wrap = np.where(diffs > np.pi)
                diffs[wrap] = 2*np.pi - diffs[wrap]
                bestMatchIndex = diffs.argmin()
                result[p][zi] = images[bestMatchIndex]

        # Save the images and plists into suitably-named folders
        for p in tqdm(range(len(targetPhases)), desc='Saving images'):
            destPath = destRoot + '/%.03lf'%targetPhases[p]
            mkdir_p(destPath)
            framesMetadata = []
            with tifffile.TiffWriter(destPath+('/%.03lf.tif'%targetPhases[p])) as tif:
                for im in result[p]:
                    tif.save(im.image.astype('uint16'))
                    framesMetadata.append(im._frameMetadata)
            SaveMetadata(framesMetadata, im._globalMetadata, im.metadataVersion, destPath+('/%.03lf.plist'%targetPhases[p]))
