'''Module for real-time phase matching of brightfield sequences.
These codes are equivalent to the Objective-C codes in spim-interface.'''

# Python Imports
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import time
import math
from copy import copy
import sys
# Local Imports
import helper as hlp
sys.path.insert(0, '../py_sad_correlation/')
import j_py_sad_correlation as jps


def updateDriftCorrection(frame0, bestMatch0, settings):
    # Assumes frame and bestMatch are numpy arrays of the same size

    dx = settings['drift'][0]
    dy = settings['drift'][1]

    # Default inset areas
    rect = [abs(dx)+1,
            frame0.shape[0]-abs(dx)-1,
            abs(dy)+1,
            frame0.shape[1]-abs(dy)-1]  # X1,X2,Y1,Y2
    bestMatch = bestMatch0[rect[0]:rect[1], rect[2]:rect[3]]

    shifts = [[0, 0],
              [1, 0],
              [-1, 0],
              [0, 1],
              [0, -1]]

    frames = np.zeros([5,
                       bestMatch.shape[0],
                       bestMatch.shape[1]],
                      dtype=frame0.dtype)

    counter = 0
    for shft in shifts:
        dxp = dx + shft[0]
        dyp = dy + shft[1]

        # Adjust for drift and shift
        rectF = copy(rect)
        rectF[0] -= dxp
        rectF[1] -= dxp
        rectF[2] -= dyp
        rectF[3] -= dyp

        frames[counter, :, :] = frame0[rectF[0]:rectF[1], rectF[2]:rectF[3]]
        counter = counter + 1

    sad = jps.sad_with_references(bestMatch, frames)
    best = np.argmin(sad)

    settings['drift'][0] = dx+shifts[best][0]
    settings['drift'][1] = dy+shifts[best][1]

    return settings


def subFrameFit(diffs, settings):
    # Find the sub-pixel position fit from a list of SADs
    # (including padding frames)
    # Initialise best and worst scores based on padding frames
    bestScore = diffs[settings['numExtraRefFrames']]
    bestScorePos = settings['numExtraRefFrames']

    # Search for lowest V
    for i in range(settings['numExtraRefFrames']+1,
                   len(diffs)-settings['numExtraRefFrames']):
        # If new lower V
        if diffs[i] < bestScore and diffs[i-1] >= diffs[i] and diffs[i+1] >= diffs[i]:
            bestScore = diffs[i]
            bestScorePos = i

    # Sub-pixel fitting
    if diffs[bestScorePos-1] < diffs[bestScorePos]:  # If no V is found
        thisFrameReferencePos = settings['numExtraRefFrames']
    else:  # Sub-pixel fitting
        interpolatedCorrection = threePointTriangularMinimum(diffs[bestScorePos-1],
                                                             diffs[bestScorePos],
                                                             diffs[bestScorePos+1])[0]
        thisFrameReferencePos = bestScorePos + interpolatedCorrection

    return thisFrameReferencePos


def threePointTriangularMinimum(y1, y2, y3):
    # Fit an even V to three points at x=-1, x=0 and x=+1
    if y1 > y3:
        x = 0.5 * (y1-y3)/(y1-y2)
        y = y2 - x * (y1-y2)
    else:
        x = 0.5 * (y1-y3)/(y3-y2)
        y = y2 + x * (y3-y2)

    return x, y


def compareFrame(frame0, referenceFrames0, settings=None, log=False, plot=False):
    # assumes frame is a numpy array and referenceFrames is a dictionary of {phase value: numpy array}

    if settings==None:
        settings = hlp.initialiseSettings()

    dx = settings['drift'][0]
    dy = settings['drift'][1]

    if plot:
        f1 = plt.figure()
        a11 = f1.add_subplot(121)
        a11.imshow(frame0)

    # Apply shifts
    if log:
        print('Applying drift correction of ({0},{1})'.format(dx, dy))
    rectF = [0,
             frame0.shape[0],
             0,
             frame0.shape[1]]  # X1,X2,Y1,Y2
    rect = [0,
            referenceFrames0[0].shape[0],
            0,
            referenceFrames0[0].shape[1]]  # X1,X2,Y1,Y2

    if dx <= 0:
        rectF[0] = -dx
        rect[1] = rect[1]+dx
    else:
        rectF[1] = rectF[1]-dx
        rect[0] = dx
    if dy <= 0:
        rectF[2] = -dy
        rect[3] = rect[3]+dy
    else:
        rectF[3] = rectF[3]-dy
        rect[2] = +dy

    frame = frame0[rectF[0]:rectF[1], rectF[2]:rectF[3]]
    referenceFrames = referenceFrames0[:, rect[0]:rect[1], rect[2]:rect[3]]

    if plot:
        a12 = f1.add_subplot(122)
        a12.imshow(frame)
        plt.show()

    # Calculate SADs
    SADs = jps.sad_with_references(frame, referenceFrames)

    if log:
        pprint(SADs)
    if plot:
        f2 = plt.figure()
        a21 = f2.add_axes([0, 0, 1, 1])
        a21.plot(range(len(SADs)), SADs)
        plt.show()

    # Identify best match
    phase = subFrameFit(SADs, settings)
    if log:
        print('Found frame phase to be {0}'.format(phase))

    # Update drift
    settings = updateDriftCorrection(frame0,
                                     referenceFrames0[np.argmin(SADs)],
                                     settings)
    if log:
        print('Drift correction updated to ({0},{1})'.format(dx, dy))

    # Note: still includes padding frames (on purpose)
    return (phase, SADs, settings)


def predictTrigger(frameSummaryHistory,
                   settings,
                   fitBackToBarrier=True,
                   log=False,
                   output="seconds"):
    # frameSummaryHistory is an nx3 array of [timestamp, phase, argmin(SAD)]
    # phase (i.e. frameSummaryHistory[:,1]) should be cumulative 2Pi phase
    # targetSyncPhase should be in [0,2pi]

    if frameSummaryHistory.shape[0] < settings['minFramesForFit']+1:
        if log:
            print('Fit failed due to too few frames...')
        return -1
    if fitBackToBarrier:
        allowedToExtendNumberOfFittedPoints = False
        framesForFit = min(settings['frameToUseArray'][int(frameSummaryHistory[-1, 2])],
                           frameSummaryHistory.shape[0])
        if log:
            print('Consider {0} past frames for prediction;'.format(framesForFit))
    else:
        framesForFit = settings['minFramesForFit']
        allowedToExtendNumberOfFittedPoints = True

    pastPhases0 = frameSummaryHistory[-int(framesForFit):, :]

    # Problem with below linear fit algorithm resulting in incorrect current phase and incorrect trigger times
    #alpha, radsPerSec = linearFit(pastPhases0[:, 0], pastPhases0[:, 1])
    radsPerSec, alpha = np.polyfit(pastPhases0[:,0],pastPhases0[:,1],1)

    if log:
        print('Linear fit with intersect {0} and gradient {1}'.format(alpha, radsPerSec))
    if log and radsPerSec < 0:
        print('Linear fit to unwrapped phases is negative! This is a problem (fakeNews).')
    elif log and radsPerSec == 0:
        print('Linear fit to unwrapped phases is zero! This will be a problem for prediction (divByZero).')

    thisFramePhase = alpha + frameSummaryHistory[-1, 0]*radsPerSec
    multiPhaseCounter = thisFramePhase//(2*math.pi)
    phaseToWait = settings['targetSyncPhase'] + (multiPhaseCounter*2*math.pi) - thisFramePhase
    # c.f. lines 1798-1801 in SyncAnalyzer.mm
    # essentially this fixes for small drops in phase due to SAD errors
    while phaseToWait < 0:  # this used to be -math.pi
        phaseToWait += 2*math.pi

    timeToWaitInSecs = phaseToWait / radsPerSec
    timeToWaitInSecs = max(timeToWaitInSecs, 0.0)

    if log:
        print('Current time: {0};\tTime to wait: {1};'.format(frameSummaryHistory[-1,0], timeToWaitInSecs))
        print('Current phase: {0};\tPhase to wait: {1};\nTarget phase:{2};\tPredicted phase:{3};'.format(thisFramePhase, phaseToWait, settings['targetSyncPhase'] + (multiPhaseCounter*2*math.pi), thisFramePhase+phaseToWait))

    frameInterval = 1.0/settings['framerate']
    if allowedToExtendNumberOfFittedPoints and timeToWaitInSecs > (settings['extrapolationFactor'] * settings['minFramesForFit'] * frameInterval):
        settings['minFramesForFit'] *= 2
        if settings['minFramesForFit'] <= pastPhases.shape[0] and settings['minFramesForFit'] <= settings['maxFramesForFit']:
            print('Increasing number of frames to use')
            timeToWaitInSecs = predictTrigger(pastPhases, settings['targetSyncPhase'])
        settings['minFramesForFit'] = settings['minFramesForFit']//2

    if output == "seconds":
        return timeToWaitInSecs
    elif output == "phases":
        return phaseToWait
    else:
        print("What are ye on mate!")
        return 0.0


def linearFit(X, Y):
    ln = len(X)
    Xm = 0
    Ym = 0
    X2m = 0
    XYm = 0
    for i in range(ln):
        Xm += X[i]
        Ym += Y[i]
        X2m += X[i]**2
    Ym = Ym/ln
    X2m = X2m/ln
    XYm = XYm/ln
    beta = (XYm-(Xm*Ym)) / (X2m-(Xm**2))
    alpha = Ym-(beta*Xm)

    return alpha, beta


def deduceBarrierFrameArray(settings):
    # frames to consider based on reference point and no padding
    numToUseNoPadding = list(range(settings['referenceFrameCount']-(2*settings['numExtraRefFrames'])))
    numToUseNoPadding = np.asarray(numToUseNoPadding[-int(settings['barrierFrame']-3):] + numToUseNoPadding[:-int(settings['barrierFrame']-3)])

    # consider padding by setting extra frames equal to last/first unpadded number
    numToUsePadding = numToUseNoPadding[-1]*np.ones(settings['referenceFrameCount'])
    numToUsePadding[settings['numExtraRefFrames']:(settings['referenceFrameCount']-settings['numExtraRefFrames'])] = numToUseNoPadding

    # consider min and max number of frames to use
    numToUsePadding = np.maximum(numToUsePadding,
                                 settings['minFramesForFit']*np.ones(settings['referenceFrameCount']))
    numToUsePadding = np.minimum(numToUsePadding,
                                 settings['maxFramesForFit']*np.ones(settings['referenceFrameCount']))

    # update settings
    settings['frameToUseArray'] = numToUsePadding
    return settings


def gotNewSyncEstimateTimeDelay(timestamp, timeToWaitInSeconds, settings, log=False):
    sendIt = 0
    framerateFactor = 1.6

    if timeToWaitInSeconds < settings['predictionLatency']:
        # too close but if not sent this period then give it a shot
        if settings['lastSent'] < timestamp-(settings['referencePeriod']/settings['framerate']):
            sendIt = 1
        else:  # if already sent this period, start prediction for next cycle
            timeToWaitInSeconds += (settings['referencePeriod']/settings['framerate'])
    elif (timeToWaitInSeconds-(framerateFactor/settings['framerate'])) < settings['predictionLatency']:
        # won't be able to do another calculation so trigger
        sendIt = 2
    else:
        pass

    if sendIt > 0 and settings['lastSent'] > (timestamp-((settings['referencePeriod']/settings['framerate'])/2)):
        # if we've done any triggering in the last half a cycle, don't trigger
        # this is quite different from JTs approach
        # where he follows which cycle we last triggered on
        if log:
            print('Trigger type {0} at {1}\tDROPPED'.format(sendIt,
                                                            timestamp+timeToWaitInSeconds))
        sendIt = 0
    elif sendIt > 0:
        settings['lastSent'] = timestamp

    return timeToWaitInSeconds, sendIt, settings


if __name__ == "__main__":
    sys.path.insert(0, 'j_postacquisition/')
    import image_loading as jil

    log = True
    t0 = time.time()

    # BEGIN SETTINGS #
    # Hardcode default Settings
    settings = hlp.initialiseSettings(drift=[-5, -2],
                                      predictionLatency=0.01,
                                      referencePeriod=143.52867145530271,
                                      framerate=80)

    # load sequence and references frames
    sequenceName = '/home/usr/Documents/Summer_Project/cjn-python-emulator/Stack 0001/Brightfield - Prosilica/'
    referenceName = '/home/usr/Documents/Summer_Project/cjn-python-emulator/Stack 0001/ref-frames-2018-07-26-10.59.26'
    # END SETTINGS #

    # Load referenceFrames
    referenceObj = jil.LoadAllImages(referenceName, True, 1, 0, -1, None)
    referenceFrames, idx = hlp.convertObj(referenceObj)

    # load sequence
    sequenceObj = jil.LoadAllImages(sequenceName, True, 1, 0, -1, None)
    sequence, idxS = hlp.convertObj(sequenceObj)
    # Still needed for plist information: est_framerate, timestamp
    sequenceObj = sequenceObj[0]

    # reference sequence details
    # includes update of referenceFrameCount and targetSyncPhase
    settings = hlp.updateSettings(settings,
                                  referencePeriod=float(sequenceObj[idxS[0]].frameKeyPath('sync_info.reference_period')),
                                  referenceFrame=sequenceObj[idxS[0]].frameKeyPath('sync_info.sync_settings.reference_frame'),
                                  barrierFrame=sequenceObj[idx[0]].frameKeyPath('sync_info.sync_settings.barrier_frame'))
    print('Reference sequence covers a period of {0} with {1} frames (including padding)'.format(settings['referencePeriod'], settings['referenceFrameCount']))

    # numFramesToUse array
    settings = deduceBarrierFrameArray(settings)

    # Set-up
    pastPhases = np.zeros([min(len(sequence),
                           settings['maxReceivedFramesToStore']),
                           3])  # timestamp,phase,lowestSADframe
    triggers = []
    times = []
    phases = []
    count = 0
    pp_old = 0  # we should never actually need this but it's there for style

    for i in range(len(sequence)):
        if log:
            print('Frame index: {0}'.format(i))
        if i >= settings['maxReceivedFramesToStore']:
            pastPhases[0:-1, :] = pastPhases[1:, :]
            j = -1
        else:
            j = i
        pp, dd, settings = compareFrame(sequence[i],
                                        referenceFrames,
                                        settings,
                                        log,
                                        False)
        # convert phase to 2pi base
        pp = ((pp-settings['numExtraRefFrames'])/settings['referencePeriod'])*(2*math.pi)
        # tt = sequenceObj[idxS[i]].plist['timestamp']
        tt = sequenceObj[idxS[i]].timestamp
        times.append(tt)

        pastPhases[j, 0] = tt
        if j != 0:  # make phase cumulative
            deltaPhase = pp-pp_old
            # c.f. lines 1798-1801 in SyncAnalyzer.mm
            # Allows for small drops in phase due to SAD errors
            while deltaPhase < -math.pi:
                deltaPhase += 2*math.pi
            pastPhases[j, 1] = pastPhases[j-1, 1]+deltaPhase
        else:
            pastPhases[j, 1] = pp
        phases.append(pastPhases[j, 1])

        pastPhases[j, 2] = np.argmin(dd)

        # Don't predict if not done a whole period
        if i > settings['referencePeriod']:
            tr = predictTrigger(np.copy(pastPhases[:i+1, :]),
                                settings,
                                True,
                                log)
            tr, send, settings = gotNewSyncEstimateTimeDelay(tt,
                                                             tr,
                                                             settings,
                                                             log)
            if send > 0:
                count = count + 1
                print('Trigger type {0} at {1}\tSENT'.format(send, tt+tr))
        else:
            tr = -1

        triggers.append(tr)
        pp_old = float(pp)
    print(count)
    print(time.time()-t0)
