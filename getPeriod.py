## Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
import tifffile as tif

from pprint import pprint
import time
import math
import os

import sys
sys.path.insert(0, '../j_postacquisition/')
import image_loading as jid
import plist_wrapper as plw
sys.path.insert(0, '../py_sad_correlation/')
import j_py_sad_correlation as jps

import realTimeSync as rts
import helper as hlp

def doEstablishPeriodProcessingForFrame(sequence,settings,padding=True,log=False):
    # Do the processing required for the initial establishing of the heart period (and associated selection of one period's worth of reference frames). Establish the period for the cycle leading up to this current frame.
    # Assumes sequence is a list of numpy arrays
    # This returns the frames (c.f. doEstablishPeriodProcessingForFrameIdx)
    referenceFrameIdx, settings = doEstablishPeriodProcessingForFrameIdx(sequence,settings,padding,log)
    #print(sequence[referenceFrameIdx].shape)
    return sequence[referenceFrameIdx], settings

def doEstablishPeriodProcessingForFrameIdx(sequence,settings,padding=True,log=False):
    # Do the processing required for the initial establishing of the heart period (and associated selection of one period's worth of reference frames). Establish the period for the cycle leading up to this current frame.
    # Assumes sequence is a list of numpy arrays

    periods = []
    for i in range(1,len(sequence)):
        frame = sequence[i,:,:]
        pastFrames = sequence[:(i-1),:,:]
        if log:
            print('Running for frame {0}'.format(i))

        # Calculate Diffs
        diffs = jps.sad_with_references(frame, pastFrames)

        # Calculate Period for Diffs
        period = calcPeriodForDiffs(diffs,log)
        if period!=-1:
            periods.append(period)

        # Get One Periods Frames
        if period!=-1 and len(periods)>=(5+(2*settings['numExtraRefFrames'])) and period>6 and (len(periods) - 1 - settings['numExtraRefFrames'])>0 and (periods[len(periods) - 1 - settings['numExtraRefFrames']])>6:
            if log:
                print('Found a period I\'m happy with')
            periodToUse = periods[len(periods)-1-settings['numExtraRefFrames']]

            settings = hlp.updateSettings(settings,referencePeriod=periodToUse)#automatically does referenceFrameCount an targetSyncPhase
            if padding:
                numRefs = math.ceil(periodToUse)+(2*settings['numExtraRefFrames'])
                return np.arange(len(pastFrames)-numRefs,len(pastFrames)), settings
            else:
                numRefs = math.ceil(periodToUse)+settings['numExtraRefFrames']
                return np.arange(len(pastFrames)-numRefs,len(pastFrames)-settings['numExtraRefFrames']), settings

    print('ERROR: I didn\'t find a period I\'m happy with!')
    return None, settings

def calcPeriodForDiffs(diffs,log=False):
    # Calculate the heart period (with sub-frame interpolation) based on a provided list of comparisons between the current frame and previous frames. The list is in reverse order (i.e. difference with most recent frame comes first)
    bestMatchPeriod = EstimatePeriod(diffs,log)
    bestMatchEntry = diffs.size - bestMatchPeriod

    if bestMatchPeriod==-1:
        return -1

    interpolatedMatchEntry = bestMatchEntry+rts.threePointTriangularMinimum(diffs[bestMatchEntry - 1],diffs[bestMatchEntry],diffs[bestMatchEntry + 1])[0];

    return diffs.size - interpolatedMatchEntry

def EstimatePeriod(diffs,log=False):
    # unlike JTs codes I assume 1 period with a start frame of 1
    if diffs.size<2:
        if log:
            print('WARNING: Not enough diffs, returning -1')
        return -1

    score = diffs[diffs.size-1]
    values = [score,score,score,score,score,0,1,1]#list of values needed in the gotScoreForDelta function: minScore, maxScore, totalScore, meanScore, minSinceMax, deltaForMinSinceMax, stage, numScores
    for d in range(2,diffs.size):
        #print(d)
        score = diffs[diffs.size-d]
        got,values = gotScoreForDelta(score,d,values,log)
        if got:
            return values[5]

    if log:
        print('WARNING: I didn\'t find a whole period, returning -1')
    return -1#catch if doesn't find a period

def gotScoreForDelta(score,d,values,log=False):
    #values = (minScore, maxScore, totalScore, meanScore, minSinceMax, deltaForMinSinceMax, stage, numScores)

    values[2] += score#totalScore
    values[7] += 1#numScores

    lowerThresholdScore = values[0] + (values[1] - values[0]) / 2#minScore,maxScore
    upperThresholdScore = values[0] + (values[1] - values[0]) *3 / 4#minScore,maxScore
    if log:
        #print('Lower Threshold:\t{0:.4f};\tUpper Threshold:\t{1:.4f}'.format(lowerThresholdScore,upperThresholdScore))
        pass

    if score<lowerThresholdScore and values[6]==1:#stage
        if log:
            print('Stage 1: Under lower threshold; Moving to stage 2')
        values[6] = 2#stage

    if score>upperThresholdScore and values[6]==2:#stage
        #TODO: speak to JT about the 'final condition'
        if log:
            print('Stage 2: Above upper threshold; Returning period of {0}'.format(values[5]))
        values[6] = 3#stage
        return True,values

    if score>values[1]:#maxScore
        if log:
            print('New max score: {0} > {1}...'.format(score,values[1]))
            print('Resetting to stage 1...')
        values[1] = score#maxScore
        values[4] = score#minSinceMax
        values[5] = d#deltaForMinSinceMax
        values[6] = 1#stage
    elif score!=0 and (values[0]==0 or score<values[0]):#minScore
        if log:
            print('New minimum score of {0}'.format(score))
        values[0] = score#minScore

    if score<values[4]:#minSinceMax
        if log:
            print('New minimum score ({0}) since maximum of {1}'.format(score,values[1]))
        values[4] = score#minSinceMax
        values[5] = d#deltaForMinSinceMax

    # Note this is only updated AFTER we have done the other processing (i.e. the mean score used does NOT include the current delta)
    values[3] = values[2] / values[7];#meanScore,totalScore,numeScores

    return False,values

def saveReferencePeriod(folder,sequenceObj,referenceIdx,period,drift=[0,0],target=0.0):
    # Compatable with ImageClass_v3
    # as such, does not save plist anymore
    # and I havge yet to work out the best way to save metadata at all
    sequence, dummy = hlp.convertObj(sequenceObj)
    if not os.path.exists(folder):
        os.makedirs(folder)
    frameNumber = sequenceObj[referenceIdx[0]].frameKeyPath('frame_number')
    tif.imsave('{0}/{1:03d}.tif'.format(folder,frameNumber),sequence[referenceIdx],metadata={'axes':'TXY'})
    with open('{0}/period.txt'.format(folder), 'w') as text_file:
        text_file.write(str(period)+' '+str(drift[1])+' '+str(drift[0])+' '+str(target))

def saveReferencePeriodOld(folder,sequenceObj,referenceIdx,period):
    # For compatability with ImageClass_v<lt3>
    if not os.path.exists(folder):
        os.makedirs(folder)
    for idx in referenceIdx:
        plst = sequenceObj[0][idx].plist
        frameNumber = plst['frame_number']
        tif.imsave('{0}/{1:03d}.tif'.format(folder,frameNumber),sequenceObj[0][idx].image,metadata={'axes':'XY'})
        plst['sync_info']['reference_period'] = period
        plw.writePlist(plst, '{0}/{1:03d}.plist'.format(folder,frameNumber))
    with open('{0}/period.txt'.format(folder), 'w') as text_file:
        text_file.write(str(period))

if __name__ == '__main__':
    settings = hlp.initialiseSettings(drift=[-5,-2],
                            framerate=80,
                            referencePeriod=42.410156325856882,
                            barrierFrame=4.9949327669365964,
                            predictionLatency=0.01,
                            referenceFrame=20.994110773833857)

    # load sequence
    sequenceName = '../notebooks/localdata/2017-02-09 18.35.13 vid overnight/Stack 0004/Brightfield - Prosilica/'
    sequenceObj = jid.LoadAllImages(sequenceName,True,1,0,-1,None)
    sequence, idx = hlp.convertObj(sequenceObj)

    # Get Period and Reference Frames from Sequence
    referenceFrameIdx, settings = doEstablishPeriodProcessingForFrameIdx(sequence,settings,log=True)

    pprint(referenceFrameIdx)
    print(len(referenceFrameIdx))

    saveReferencePeriod('./localdata',sequenceObj,referenceFrameIdx,settings['referencePeriod'])
