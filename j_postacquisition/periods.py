import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from math import log, sqrt, sin, floor
import scipy.signal, scipy.ndimage
import sys, time, warnings
from tqdm import *
from image_class import *

def ScoreCandidatePeriod(images, period, k):
    # Phase-wrap the frame indices

    # Establish an array whose columns represent:
    # - array index
    # - frame index
    # - phase-wrapped frame index
    a = np.zeros((len(images), 3))
    a[:,0] = np.arange(len(images))
    a[:,1] = np.vectorize(ImageClass.fi)(images)
    a[:,2] = a[:,1] - period * (a[:,1] / period).astype('int')
    # Sort by phase-wrapped frame index
    a = a[a[:,2].argsort(kind='mergesort')]     # Using mergesort as it is a stable sort

    # Calculate the string length as per Liebling's paper
    # Between each adjacent element we calculate sqrt(diff^2 + deltaT^2)
    first = a[0:len(images), 0].astype('int')
    second = a[1:, 0].astype('int')
    second = np.append(second, a[0, 0].astype('int'))
    deltaT = a[1:, 2] - a[0:len(images)-1, 2]
    deltaT = np.append(deltaT, a[0, 2] + period - a[len(images)-1, 2])
    score = 0
    for i in range(0, len(first)):
        score = score + np.sum((images[first[i]].image - images[second[i]].image)**2 + k*(deltaT[i]/float(period))**2)

    return score

def ScorePeriodsForImageSequence(images, k, periodRange):
    scores = []
    for period in periodRange:
        scores.append(ScoreCandidatePeriod(images, period, k))
    return np.array(scores)

def EstablishPeriodForImageSequence(images, periodRange, plotAllPeriods=False):
    # It seems that k=1e3 to 1e4 is the right ballpark for 80fps brightfield images,
    # in that that's when the graph starts to change noticably.
    # TODO: I should see if that needs improving or generalizing, though
    if (periodRange is None):
        return None
    scores = ScorePeriodsForImageSequence(images, k=1000, periodRange=periodRange)
    if plotAllPeriods:
        plt.plot(periodRange, scores)
        plt.show()
    return periodRange[np.argmin(scores)]

def Interpolate(a, b, frac):
    return a * (1 - frac) + b * frac

def PlotIntensitiesToIdentifyLaserFlashes(images):
    # To help identify a threshold to use for filtering out laser flashes from a brightfield dataset,
    # plot a graph of intensity against frame number
    intens = []
    for i in range(len(images)):
        intens.append(images[i].image.sum())
    plt.plot(intens)
    plt.show()

def ResampleImageSection(imageSection, sectionPeriod, numSamplesPerPeriod, numPeriodsToUse, intensityThresholdForExclusion = None):
    # See comments below that give some info about how intensityThresholdForExclusion works
    result = []

    assert(len(imageSection) >= numPeriodsToUse * sectionPeriod)

    # Optional: discard frames where the intensity is highly anomalous, since these (corresponding to laser flashes)
    # are likely to mess up the alignment calculations.
    intens = []
    for i in range(len(imageSection)):
        intens.append(imageSection[i].image.sum())
    intens = np.array(intens)

    if (intensityThresholdForExclusion is not None):
        # Mark images whose intensity is more than a critical value above the average.
        # Note that by "average" I mean the average summed-intensity across all the images
        # in *this specific image section* we have been passed to process.
        # This is intended to filter out brightfield frames contaminated by a laser pulse.
        # The function PlotIntensitiesToIdentifyLaserFlashes can be useful for determining
        # the appropriate threshold for a given dataset.
        avg = np.average(intens)
        anomalous = (intens > avg + intensityThresholdForExclusion)
        anomalousCount = np.count_nonzero(anomalous)
    else:
        anomalousCount = 0

    for i in range(numPeriodsToUse * numSamplesPerPeriod):
        desiredPos = (i / float(numSamplesPerPeriod)) * sectionPeriod
        beforePos = int(desiredPos)
        afterPos = beforePos + 1
        if (intensityThresholdForExclusion is not None):
            # If this frame has been marked as an anomalous one, choose the one before/after it
            # to use in our interpolation instead.
            # Note that I have not handled the case where the very first or last image
            # in our sequence is the anomalous one. In that case we just retain the
            # anomaly. Obviously this is not ideal, and could be improved, but the
            # current situation is a lot better than not filtering out anything at all!
            # In practice, I suspect we would only have problems if *both* of
            # a sequence pair that we are comparing have a laser flash that escaped filtering.
            while (anomalous[beforePos] and beforePos > 0):
                beforePos = beforePos - 1
            while (anomalous[afterPos] and afterPos < len(imageSection)-1):
                afterPos = afterPos + 1
        before = imageSection[beforePos]
        after = imageSection[afterPos%len(imageSection)]
        remainder = (desiredPos - beforePos) / float(afterPos - beforePos)
        assert(remainder >= 0)
        assert(remainder < 1)

        image = ImageClass()
        image.image = Interpolate(before.image, after.image, remainder)
        image.frameIndex = Interpolate(before.frameIndex, after.frameIndex, remainder)
        image.timestamp = Interpolate(before.timestamp, after.timestamp, remainder)
        result.append(image)
    return (result, anomalousCount)

def ResampleUniformly(imageSections, sectionPeriods, numSamplesPerPeriod, numPeriodsToUse = 2, intensityThresholdForExclusion = None):
    # Resample each period into exactly the same number of (interpolated) frames per period
    t1 = time.time()
    resampledImageSections = []
    anomalousCount = 0
    totalCount = 0

    for i in tqdm(range(len(imageSections)), desc=('resample, using %d samples per period' % numSamplesPerPeriod)):
        (resampled, thisAnomalousCount) = ResampleImageSection(imageSections[i], sectionPeriods[i], numSamplesPerPeriod, numPeriodsToUse, intensityThresholdForExclusion)
        resampledImageSections.append(resampled)
        anomalousCount = anomalousCount + thisAnomalousCount
        totalCount = totalCount + len(imageSections[i])
    print(anomalousCount, '/', totalCount, 'skipped as anomalous')
    return resampledImageSections

def SplitIntoSections(images, averagePeriod, periodRange, plotAllPeriods=False, alpha=10, numPeriodsToUse=2):
    # Split up the sequence into sections of 2T+alpha in length
    # (discarding any remainder at the end of the sequence)
    maxSectionLength = floor(numPeriodsToUse * averagePeriod + alpha)
    imageSections = []
    sectionPeriods = []

    start = 0

    #    for i in tqdm(range(int(len(images) / sectionLength)), desc='Dividing and calculating periods'):
    with tqdm(total=len(images), desc='Dividing and calculating periods') as pbar:
        while (start + maxSectionLength < len(images)):
            thisSection = images[start : start+maxSectionLength]
            thisPeriod = EstablishPeriodForImageSequence(thisSection, periodRange, plotAllPeriods=plotAllPeriods)
            lenToUse = int(thisPeriod*numPeriodsToUse)+1
            #print (start, lenToUse, thisPeriod, len(thisSection))
            assert(lenToUse <= len(thisSection))
            imageSections.append(thisSection[0:lenToUse+1]) # +1 because of the way ranges go from [start:end+1]
            sectionPeriods.append(thisPeriod)
            start += lenToUse
            del thisSection
            pbar.update(lenToUse)
        pbar.update(len(images) - start)

    print('section periods', sectionPeriods)
    return (imageSections, sectionPeriods)
