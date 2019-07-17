# Calculates relative timeshifts between pairs of image sequences that, when applied, synchronize the sequences
# (i.e. both start at the same heartbeat phase)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from math import log, sqrt, sin
import scipy.signal, scipy.ndimage
import sys, time, warnings
from tqdm import *

def ScoreSequences(sec1, sec2, window1=None, window2=None):
    # Utility function that returns a score for the level of correlation between two sequences
    # (which have already been time-shifted, if that's what we want to do)
    diffSq = 0
    if (window1 is None):
        window1 = (0, 0, sec1[0].image.shape[0], sec1[0].image.shape[1])
    if (window2 is None):
        window2 = (0, 0, sec2[0].image.shape[0], sec2[0].image.shape[1])
    assert(len(sec1) == len(sec2))
    assert(window1[2] == window2[2])
    assert(window1[3] == window2[3])

    for i in range(len(sec1)):
        view1 = np.array(sec1[i].image[window1[0]:window1[0]+window1[2], window1[1]:window1[1]+window1[3]], copy=False)
        view2 = np.array(sec2[i].image[window2[0]:window2[0]+window2[2], window2[1]:window2[1]+window2[3]], copy=False)
        diffSq = diffSq + np.sum((view1 - view2)**2)

    return diffSq

def ScoreSequenceShift(sec1, sec2, shift, numSamplesPerPeriod, window1=None, window2=None):
    # Utility function used to score a single relative timeshift of two sequences.
    a = np.array(sec1[0:numSamplesPerPeriod])
    b = np.roll(np.array(sec2[0:numSamplesPerPeriod]), -shift)
    return ScoreSequences(a, b, window1, window2)

def MakeArrayFromSequence(seq, window):
    # Utility function used as part of the FFT processing
    if (window is None):
        window = (0, 0, seq[0].image.shape[0], seq[0].image.shape[1])
    result = np.zeros((len(seq), window[2]*window[3]))
    for i in range(len(seq)):
        result[i] = np.array(seq[i].image[window[0]:window[0]+window[2], window[1]:window[1]+window[3]], copy=False).flatten()
    return result

def MakeArrayFromSequence2(seq, window):
    # Utility function used as part of the FFT processing
    if (window is None):
        window = (0, 0, seq[0].image.shape[0], seq[0].image.shape[1])
    result = np.zeros((window[2]*window[3], len(seq)))
    for i in range(len(seq)):
        result[:,i] = np.array(seq[i].image[window[0]:window[0]+window[2], window[1]:window[1]+window[3]], copy=False).flatten()
    return result

def ShiftScoresForSequences(seqA, seqB, numSamplesPerPeriod, window1=None, window2=None, useFFT=True):
    # Get the scores for each possible relative time shift of these two specific sequences
    
    # TODO: It looks like I actually seem to consider the first period here in the FFT calc, rather than the full length of seqA and seqB.
    # I should decide if that's what I want or not.

    if (useFFT == False):
        # Slow explicit version
        scores = []
        for shift in range(0, numSamplesPerPeriod):
            scores.append(ScoreSequenceShift(seqA, seqB, shift, numSamplesPerPeriod, window1, window2))
    elif True:
        # Faster FFT-based version
        # Note that this could be speeded up significantly if I cache the fft of the input arrays
        # I can certainly do that in the absence of drift, but care may be needed if drift present,
        # depending on how exactly I handle it.
        
        # First form a pair of arrays from the image data
        t1 = time.time()
        a = MakeArrayFromSequence(seqA[0:numSamplesPerPeriod], window1)
        b = MakeArrayFromSequence(seqB[0:numSamplesPerPeriod], window2)
        t2 = time.time()
        # Now calculate the cross-correlation
        #        temp = np.conj(np.fft.rfft(a, axis=0)) * np.fft.rfft(b, axis=0)
        temp = np.conj(np.fft.fft(a, axis=0)) * np.fft.fft(b, axis=0)
        t3 = time.time()
        #temp2 = np.fft.irfft(temp, len(a[:,0]), axis=0)
        temp2 = np.fft.ifft(temp, axis=0)
        t4 = time.time()
        scores = np.sum(a*a) + np.sum(b*b) - 2 * np.sum(np.real(temp2), axis=1)
        t5 = time.time()
        #print('time breakdown', t2-t1, t3-t2, t4-t3, t5-t4)
    else:
        # I tried this to see how it performs, but doing it this way round is not really any faster
        t1 = time.time()
        a = MakeArrayFromSequence2(seqA[0:numSamplesPerPeriod], window1)
        b = MakeArrayFromSequence2(seqB[0:numSamplesPerPeriod], window2)
        t2 = time.time()
        temp = np.conj(np.fft.fft(a, axis=1)) * np.fft.fft(b, axis=1)
        t3 = time.time()
        temp2 = np.fft.ifft(temp, axis=1)
        t4 = time.time()
        scores = np.sum(a*a) + np.sum(b*b) - 2 * np.sum(np.real(temp2), axis=0)
        t5 = time.time()
        #print('time breakdown', t2-t1, t3-t2, t4-t3, t5-t4)
    
    return scores

def MakeOffsetWindow(iSize, jSize, inset, di, dj):
    return (inset+di, inset+dj, iSize-2*inset, jSize-2*inset)

def ShiftScoresForSequencesWithDrift(seqA, seqB, drift, inset, numSamplesPerPeriod, useFFT=True):
    (iSize, jSize) = seqA[0].image.shape
    window1 = MakeOffsetWindow(iSize, jSize, inset, 0, 0)
    window2 = MakeOffsetWindow(iSize, jSize, inset, drift[0], drift[1])
    return ShiftScoresForSequences(seqA, seqB, numSamplesPerPeriod, window1, window2, useFFT)

def FindMinimum(scores):
    # Given a set of difference scores, determine where the minimum lies
    if False:
        # No sub-integer interpolation
        minPos = np.argmin(scores)
        minVal = np.min(scores)
    else:
        # V-fitting for sub-integer interpolation
        # Note that 'scores' is a ring vector, with scores[0] being adjacent to scores[-1]
        y1 = scores[np.argmin(scores)-1]    # Works even when minimum is at 0
        y2 = scores[np.argmin(scores)]
        y3 = scores[(np.argmin(scores)+1) % len(scores)]
        if (y1 > y3):
            minPos = 0.5 * (y1 - y3) / (y1 - y2);
            minVal = y2 - minPos * (y1 - y2);
        else:
            minPos = 0.5 * (y1 - y3) / (y3 - y2);
            minVal = y2 + minPos * (y3 - y2);
        minPos = (minPos + np.argmin(scores)) % len(scores)

    return (minPos, minVal)


def GetShifts(resampledImageSections, sectionPeriods, sequenceDrifts, inset, numSamplesPerPeriod, maxOffsetToConsider=None, useFFT=True):
    # Determine the best possible relative alignments (time shifts) of different image sections,
    # as part of synchronization/phase assignment analysis.

    # Note: Liebling showed how this can effectively be done using FFTs.
    # I have had a go at implementing that, and I find that for my scenarios it is a little faster, but no massively so
    # I suspect that's because the layout in memory is not particularly friendly to the FFT. It might be possible to optimize
    # more by rearranging things in memory - but probably at the expense of clarity and simplicity.
    t1 = time.time()
    shifts = []
    for i in tqdm(range(0, len(resampledImageSections)), desc='calculate sequence shifts'):
        # Things will get out of control if we compare every sequence with every other, for a large dataset.
        # I am going to try making a restricted list of offsets and only calculating these.
        # They are intended to go up in powers of 2
        candidateOffsets = np.array([1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
        if (maxOffsetToConsider is not None):
            candidateOffsets = candidateOffsets[np.where(candidateOffsets <= maxOffsetToConsider)]
        offsetsToUse = i + candidateOffsets[np.where(i+candidateOffsets < len(resampledImageSections))]
        for j in offsetsToUse:
            drift = (sequenceDrifts[j][0]-sequenceDrifts[i][0], sequenceDrifts[j][1]-sequenceDrifts[i][1])
            scores = ShiftScoresForSequencesWithDrift(resampledImageSections[i], resampledImageSections[j], drift, inset, numSamplesPerPeriod, useFFT)
            (minPos, minVal) = FindMinimum(scores)
            shifts.append((i, j, minPos, minVal))
    print(' took', time.time() - t1)
    return shifts
