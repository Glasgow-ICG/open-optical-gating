# This module is intended as a helper for Spim GUI (among other things).
# It provides the information to maintain a consistent phase lock across
# multiple independent sequences (but ones with some degree of similarity, of course)

import plist_wrapper as jPlist
from image_loading import *
from image_saving import *
from shifts import *
from shifts_global_solution import *
import time
import warnings


def RoIForReferenceHistory(resampledSequences):
    # Return the shape of the reference history.
    # We presume all have the same size (caller really should ensure this, or we will run into major problems!)
    if (len(resampledSequences) == 0):
        return (-1, -1)
    return resampledSequences[0][0].image.shape


def processNewReferenceSequence(rawRefFrames,
                                thisPeriod,
                                thisDrift,
                                resampledSequences,
                                periodHistory,
                                driftHistory,
                                shifts,
                                knownPhaseIndex=0,
                                knownPhase=0,
                                numSamplesPerPeriod=80,
                                maxOffsetToConsider=2,
                                log=True):
    # Note this function has been edited to resemble cjn-sequence-alignment
    # rawRefFrames: a list of numpy arrays representing the new raw reference frames (without any "extra reference frames" as introduced by the numExtraRefFrames variable in my C code)
    # thisPeriod: the period for the frames in rawRefFrames (caller must determine this)
    # knownPhaseIndex: the index into resampledSequences for which we have a known phase point we are trying to match
    # knownPhase: the phase we are trying to match
    # periodHistory: a list of the periods for resampledSequences
    # shifts: a list of all the shifts we have already calculated within resampledSequences
    # numSamplesPerPeriod: number of samples to use in uniform resampling of the period of data
    # maxOffsetToConsider: how far apart in the resampledSequences list to make comparisons
    #                      (this should be used to prevent comparing between sequences that are so far apart they have very little similarity)

    # Deal with rawRefFrames type
    if type(rawRefFrames) is list:
        rawRefFrames = np.dstack(rawRefFrames)
        rawRefFrames = np.rollaxis(rawRefFrames,-1)

    # Check that the reference frames have a consistent shape
    for f in range(1, len(rawRefFrames)):
        if rawRefFrames[0].shape != rawRefFrames[f].shape:
            # There is a shape mismatch.
            if log:
                # Return an error message and code to indicate the problem.
                print('Error: There is shape mismatch within the new reference frames. Frame 0: {0}; Frame {1}: {2}'.format(rawRefFrames[0].shape, f, rawRefFrames[f].shape))
            return (resampledSequences,
                    periodHistory,
                    driftHistory,
                    shifts,
                    -1000.0,
                    None)
    # And that shape is compatible with the history that we already have
    if len(resampledSequences) > 1:
        if rawRefFrames[0].shape != resampledSequences[0][0].image.shape:
            # There is a shape mismatch.
            if log:
                # Return an error message and code to indicate the problem.
                print('Error: There is shape mismatch with historical reference frames. Old shape: {1}; New shape: {2}'.format(resampledSequences[0][0].shape, rawRefFrames[0].shape))
            return (resampledSequences,
                    periodHistory,
                    driftHistory,
                    shifts,
                    -1000.0,
                    None)

    # Wrap rawRefFrames in a list of ImageClass objects, since this is what my python code expects
    imageClassList = []
    for i in range(len(rawRefFrames)):
        image = ImageClass()
        image.image = rawRefFrames[i]
        image.frameIndex = i
        image.timestamp = i
        imageClassList.append(image)
    # Now resample this list of images, and add them to our sequence set
    (thisResampledSequence, anomalousCount) = ResampleImageSection(imageClassList, thisPeriod, numSamplesPerPeriod, 1)

    # Add this new sequence to the sequence history
    resampledSequences.append(thisResampledSequence)
    periodHistory.append(thisPeriod)

    if log:
        warnings.warn('No drift correction is being applied. This will seriously impact phase locking. Please use cjn-sequence-alignment.')

    # Update our shifts array, comparing the current sequence with recent previous ones
    if (len(resampledSequences) > 1):
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(resampledSequences) - maxOffsetToConsider - 1)
        for i in range(firstOne, len(resampledSequences)-1):
            # I am experimenting with using an FFT to calculate the cross-correlation faster.
            # It's a bit faster, but I can probably improve on this (I feel it should be faster still...)
            t1 = time.time()
            #            scores2 = ShiftScoresForSequencesWithDrift(resampledSequences[i], thisResampledSequence, (0,0), 0, numSamplesPerPeriod, useFFT=False)
            t2 = time.time()
            scores = ShiftScoresForSequencesWithDrift(resampledSequences[i], thisResampledSequence, (0,0), 0, numSamplesPerPeriod, useFFT=True)
            t3 = time.time()
            #print ('times', t2-t1, t3-t2)
            #print scores
            #print scores2
            #plt.plot(scores - scores2)
            #plt.show()
            if False:
                plt.plot(scores)
                plt.title(str(i)+'->'+str(len(resampledSequences)-1))
                plt.show()
            (minPos, minVal) = FindMinimum(scores)
            shifts.append((i, len(resampledSequences)-1, minPos, minVal))

    if log:
        pprint(shifts)

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = MakeShiftsSelfConsistent(shifts, len(resampledSequences), numSamplesPerPeriod, knownPhaseIndex, knownPhase)

    if log:
        print('solution:')
        pprint(globalShiftSolution)

    if log:
        print('residuals from least squares:')
        pprint(residuals)

    finalAdjacentResiduals = [];
    ssErr = 0
    for i in range(len(adjustedShifts)):
        # Compare the calculated shifts against the measured shifts for adjacent pairs
        firstIndex = adjustedShifts[i][0]
        secondIndex = adjustedShifts[i][1]
            #if ((secondIndex - firstIndex) == 1):
        calculatedShift = globalShiftSolution[secondIndex] - globalShiftSolution[firstIndex]
        measuredShift = adjustedShifts[i][2]
        #print('comparison', firstIndex, secondIndex, calculatedShift, measuredShift, calculatedShift-measuredShift)
        ssErr = ssErr + (calculatedShift - measuredShift)**2
    if (len(adjustedShifts) > 0):
        rmsErr = sqrt(ssErr / len(adjustedShifts))
        #print('final adjacent residuals:', finalAdjacentResiduals)
        if log:
            print('rms error:', rmsErr)

    # Note for developers:
    # there are two other return statements in this function
    return (resampledSequences,
            periodHistory,
            driftHistory,
            shifts,
            globalShiftSolution[-1],
            residuals)


def TestProcessingWithReferenceStacks(path, interval = 1, maxOffsetToConsider = 2, numSamplesPerPeriod = 80, knownPhaseIndex=0, knownPhase=0, exportPathForStackViewer = None):
    # Provide processNewReferenceSequence with a series of sequences previously acquired live
    fileList = []
    for file in os.listdir(path):
        if file.startswith("ref-frames"):
            fileList.append(path+'/'+file)
    fileList = sorted(fileList)

    imageSections = []
    sectionPeriods = []
    for i in range(len(fileList)):
        if ((i % interval) == 0):
            refPath = fileList[i]
            # Load full frame set, as well as period
            (images, averagePeriod) = LoadAllImages(refPath, downsampleFactor=1, periodRange=None)
            # images is an ndarray where each element is a list of ndarrays
            imageSections.append(images)
            periodPath = refPath + '/period.txt'
            with open(periodPath) as f:
                period = [float(x) for x in next(f).split()][0]
            sectionPeriods.append(period)

    resampledSequences = []
    periodHistory = []
    shifts = []
    allResultShifts = []
    for i in range(len(imageSections)):
        print ('Test with section', i)
        # Synthesize a list of numpy arrays, as our function is expecting
        rawRefFrames = []
        for j in range(len(imageSections[i])):
            rawRefFrames.append(imageSections[i][j].image)

        (resampledSequences, periodHistory, dummy, shifts, resultShift, residuals) = processNewReferenceSequence(rawRefFrames,
                                                                                                                 sectionPeriods[i],
                                                                                                                 None,
                                                                                                                 resampledSequences,
                                                                                                                 periodHistory,
                                                                                                                 None,
                                                                                                                 shifts,
                                                                                                                 knownPhaseIndex=knownPhaseIndex,
                                                                                                                 knownPhase=knownPhase,
                                                                                                                 numSamplesPerPeriod=numSamplesPerPeriod,
                                                                                                                 maxOffsetToConsider=maxOffsetToConsider)
        print ('resultShift =', resultShift, resultShift%numSamplesPerPeriod)
        print ('periodHistory =', periodHistory)
        allResultShifts.append(resultShift)
        print ('allResultShifts', allResultShifts)

    if (exportPathForStackViewer is not None):
        # For each sequence, rotate it appropriately and write out the images
        for i in range(len(imageSections)):
            thisPath = exportPathForStackViewer + ('/Seq_%03d' % i)
            os.system('mkdir -p "%s" 2>/dev/null' % thisPath)
            alignedImages = np.roll(np.array(resampledSequences[i]), -int(round(allResultShifts[i])))
            # Fill in metadata used by StackViewer
            for j in range(len(alignedImages)):
                im = alignedImages[j]
                im.metadataVersion = 1
                im._frameMetadata = { 'postacquisition_phase': (2.0 * np.pi * j / float(len(alignedImages))), \
                                        'stage_positions' : { 'last_known_z_time':j, 'last_known_z':0 }, \
                                        'time_processing_started' : j }
            # Save the images
            SaveImagesToFolder(alignedImages, thisPath, withMetadata=True)


def TestProcessing():
    # Provide processNewReferenceSequence with a series of sequences, and see what it produces for us

    # First load the sequences for ourselves
    basePath = '/Users/jonny/Movies/2016-06-30 17.38.56 vid single plane piv inflow'
    pathFormat = '%s/Brightfield - Prosilica/%06d.tif'
    periodRange = np.arange(45, 55, 0.1)
    (images, averagePeriod) = LoadImages(basePath, pathFormat, 18000, 1000, downsampleFactor=1, periodRange=None)
    averagePeriod = 50    # To save time, hard-code this approximate value
    (imageSections, sectionPeriods) = SplitIntoSections(images, averagePeriod, periodRange=periodRange, numPeriodsToUse=1)

    # Now provide them one at a time to the function we are testing
    resampledSequences = []
    periodHistory = []
    shifts = []
    numSamplesPerPeriod = 60
    maxOffsetToConsider = 2
    for i in range(len(imageSections)):
        print ('Test with section', i)
        # Synthesize a list of numpy arrays, as our function is expecting
        rawRefFrames = []
        for j in range(len(imageSections[i])):
            rawRefFrames.append(imageSections[i][j].image)

        (resampledSequences, periodHistory, dummy, shifts, resultShift, residuals) = processNewReferenceSequence(rawRefFrames,
                                                                                                                 sectionPeriods[i],
                                                                                                                 None,
                                                                                                                 resampledSequences,
                                                                                                                 periodHistory,
                                                                                                                 None,
                                                                                                                 shifts,
                                                                                                                 numSamplesPerPeriod=numSamplesPerPeriod,
                                                                                                                 maxOffsetToConsider=maxOffsetToConsider)
        print ('resultShift =', resultShift, resultShift%numSamplesPerPeriod)
        print ('periodHistory =', periodHistory)


if __name__ == "__main__":
    TestProcessingWithReferenceStacks('/home/chas/data/2017-01-26 Panna overnight - refs only', interval=1, maxOffsetToConsider=3, exportPathForStackViewer=None)
    #if False:
        #TestProcessingWithReferenceStacks('/Users/jonny/Movies/2016-07-25 long term sync PARTIAL/refs_only', interval=2, maxOffsetToConsider=5, exportPathForStackViewer='/Users/jonny/Movies/2016-07-25 long term sync PARTIAL/aligned')
    #elif True:
        #TestProcessingWithReferenceStacks('/Volumes/Jonny Data Work/spim_vids_2/2017-01-26 Panna overnight/Run 2 PARTIAL/refs_only', interval=1, maxOffsetToConsider=4, exportPathForStackViewer='/Users/jonny/Movies/Panna temp analysis')
#    TestProcessing()
