import image_loading as jil
import tifffile as tif
import numpy as np
import asciiplotlib as apl
import os
import sys
sys.path.insert(0, '../cjn-python-emulator/')
import helper as hlp


def convertToMultiPageTiff(rootFolder,folderBegins=None,outFolder=None,frameIndexStart=0,frameIndexFromEnd=0):
    # Find *all* folders to convert
    directoryList = []
    for root, directories, filenames in os.walk(rootFolder):
        for directory in directories:
            if (folderBegins is not None) and (directory.startswith(folderBegins)):
                directoryList.append(os.path.join(root, directory))
            elif (folderBegins is None):
                directoryList.append(os.path.join(root, directory))
    directoryList = sorted(directoryList)
    print(directoryList)

    # Covert images in each folder
    for directory in directoryList:
        fileList = []
        for f in sorted(os.listdir(directory)):
            if (f.endswith(".tif") or f.endswith(".tiff")):
                fileList.append(os.path.join(directory,f))
        if frameIndexStart>0:
            fileList = fileList[frameIndexStart:]
        if frameIndexFromEnd>0:
            fileList = fileList[:-frameIndexFromEnd]
        images = jil.LoadOrCacheImagesFromFileList(fileList, loadImageData=True, downsampleFactor=1, frameIndexStart=0, periodRange=None, plotAllPeriods=False, cropRect=None, cropFollowsZ=None, timestampKey='time_processing_started')
        if outFolder is None:
            outFolder = directory
        relative = os.path.relpath(directory,os.path.commonpath([rootFolder,directory]))
        outName = relative.replace('/','_')+'.tif'
        print('Converting...')
        images, idx = hlp.convertObj(images)
        print('Saving to {0}...'.format(os.path.join(outFolder,outName)))
        tif.imsave(os.path.join(outFolder,outName),images,metadata={'axes':'ZXY'})

    print('Fin')


def mips(im, orientations):
    '''mips creates X, Y or Z maximum intensity projections
    * image must be an PxMxN or 3xPxMxN numpy array
    * orientations a string indicating projection axis,
      e.g. any of ['Z','Y','X'] or ['xy','xz','yz']'''

    mip = None  # in case no orientations are correctly requested

    # Check valid image size (assume numpy array)
    if im.ndim == 3:
        print('Assuming Greyscale 3D Image...')
        rgb = False
    elif im.ndim == 4 and im.shape[0] == 3:
        print('Assuming RGB 3D Image...')
        rgb = True
    else:
        print('Seemingly invalid...')
        return mip

    # Create Z-projections
    if orientations == 'Z' or orientations == 'xy':
        if rgb:
            mip = im.max(axis=1)
        else:
            mip = im.max(axis=0)

    # Create Y-projections
    if orientations == 'Y' or orientations == 'xz':
        if rgb:
            mip = im.max(axis=3)
        else:
            mip = im.max(axis=2)

    # Create X-projections
    if orientations == 'X' or orientations == 'yz':
        if rgb:
            mip = im.max(axis=2)
        else:
            mip = im.max(axis=1)

    return mip


def fuse(r=[0], g=[0], b=[0]):
    '''fuse creates an RGB (3xPxMxN) numpy array from greyscale images
    * r,g,b are PxMxN numpy arrays representing greyscale images
      P may equal 1'''

    r = np.asarray(r, dtype='uint16')
    g = np.asarray(g, dtype='uint16')
    b = np.asarray(b, dtype='uint16')

    # Determine P
    dim = max([r.ndim, g.ndim, b.ndim])

    # Find biggest shape needed
    if dim == 2:
        sz = np.asarray([1, 1])
        sz = np.maximum(sz, r.shape)
        sz = np.maximum(sz, g.shape)
        sz = np.maximum(sz, b.shape)
    elif dim == 3:
        sz = np.asarray([1, 1, 1])
        sz = np.maximum(sz, r.shape)
        sz = np.maximum(sz, g.shape)
        sz = np.maximum(sz, b.shape)

    # Convert all channels to 1x[Px]MxN
    if len(sz) == 3:
        sz = [1, sz[0], sz[1], sz[2]]
    elif len(sz) == 2:
        sz = [1, sz[0], sz[1]]
    r = np.resize(r, sz)
    g = np.resize(g, sz)
    b = np.resize(b, sz)

    # Concatenate into RGB image
    rgb = np.concatenate([r, g, b], axis=0)

    return rgb


def processTimelapse(root,
                     stack='Stack {0:04d}',
                     channelR=None,
                     channelG=None,
                     channelB=None,
                     stackRange=[1]):
    if channelR is not None:
        print('Checking Red Channel Slice Numbers:')
        checkSliceNumber(root, channelR)
    if channelG is not None:
        print('Checking Green Channel Slice Numbers:')
        checkSliceNumber(root, channelG)
    if channelB is not None:
        print('Checking Blue Channel Slice Numbers:')
        checkSliceNumber(root, channelB)

    for stackNumber in stackRange:
        if channelR is not None:
            imR, dummy = jil.LoadAllImages(os.path.join(root,
                                                        stack.format(stackNumber),
                                                        channelR),
                                           periodRange=None)
            imR, dummy = hlp.convertObj(imR)
            ZmipR = mips(imR, 'xy')
            tif.imsave(os.path.join(root,
                                         'XY Mip',
                                         channelR,
                                         '{0:04d}.tif'.format(stackNumber)),
                            ZmipR)
            YmipR = mips(imR, 'xz')
            tif.imsave(os.path.join(root,
                                         'XZ Mip',
                                         channelR,
                                         '{0:04d}.tif'.format(stackNumber)),
                            YmipR)
        else:
            ZmipR = [0]
            YmipR = [0]

        if channelG is not None:
            imG, dummy = jil.LoadAllImages(os.path.join(root,
                                                        stack.format(stackNumber),
                                                        channelG),
                                           periodRange=None)
            imG, dummy = hlp.convertObj(imG)
            ZmipG = mips(imG, 'xy')
            tif.imsave(os.path.join(root,
                                         'XY Mip',
                                         channelG,
                                         '{0:04d}.tif'.format(stackNumber)),
                            ZmipG)
            YmipG = mips(imG, 'xz')
            tif.imsave(os.path.join(root,
                                         'XZ Mip',
                                         channelG,
                                         '{0:04d}.tif'.format(stackNumber)),
                            YmipG)
        else:
            ZmipG = [0]
            YmipG = [0]

        if channelB is not None:
            imB, dummy = jil.LoadAllImages(os.path.join(root,
                                                        stack.format(stackNumber),
                                                        channelB),
                                           periodRange=None)
            imB, dummy = hlp.convertObj(imB)
            ZmipB = mips(imB, 'xy')
            tif.imsave(os.path.join(root,
                                         'XY Mip',
                                         channelB,
                                         '{0:04d}.tif'.format(stackNumber)),
                            ZmipB)
            YmipB = mips(imB, 'xz')
            tif.imsave(os.path.join(root,
                                         'XZ Mip',
                                         channelB,
                                         '{0:04d}.tif'.format(stackNumber)),
                            YmipB)
        else:
            ZmipB = [0]
            YmipB = [0]

        Zrgb = fuse(ZmipR, ZmipG, ZmipB)
        tif.imsave(os.path.join(root,
                                     'XY Mip',
                                     'fused',
                                     '{0:04d}.tif'.format(stackNumber)),
                        Zrgb)
        Yrgb = fuse(YmipR, YmipG, YmipB)
        tif.imsave(os.path.join(root,
                                     'XZ Mip',
                                     'fused',
                                     '{0:04d}.tif'.format(stackNumber)),
                        Yrgb)


def checkSliceNumber(rootFolder, channelName):
    # Get all subfolders
    directoryNames = sorted([x for x in os.listdir(rootFolder)])
    # remove Mip folders
    directoryNames = [x for x in directoryNames if (x.startswith('Stack'))]

    # Run through all stacks, counting slices
    stacks = []
    numberSlices = []

    for stack in directoryNames:
        print('Counting {0}...'.format(os.path.join(rootFolder,
                                                    stack,
                                                    channelName)))
        stackName = stack.split()
        stackNumber = int(float(stackName[-1]))
        stacks.append(stackNumber)

        images, dummy = jil.LoadAllImages(os.path.join(rootFolder,
                                                       stack,
                                                       channelName),
                                          loadImageData=False,
                                          periodRange=None,
                                          saveCache=False)
        numberSlices.append(len(images))
        print('Counted {0} slices'.format(len(images)))

    f = apl.figure()
    f.plot(stacks, numberSlices)
    f.show()


if __name__ == "__main__":
    convertToMultiPageTiff('/home/chas/data/2017-02-09 18.35.13 vid overnight development',
                           folderBegins='Red',
                           outFolder='/home/chas/data/2017-02-09 multipage')
    # root = '/home/chas/data/2018-06-neutrophils/RAW'
    # processTimelapse(root,
    #                  channelR='mcherry - QIClick Q35977',
    #                  channelG='Green - QIClick Q35977',
    #                  stackRange=np.arange(1, 461))
