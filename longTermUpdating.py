## Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pprint import pprint
import time
import math
import os
from copy import deepcopy

import sys
sys.path.insert(0, 'j_postacquisition/')
import image_loading as jil

import realTimeSync as rts
import helper as hlp


def emulateSequence(sequenceObj,referenceFrames,settings,update=False,log=False,timestamp='time_processing_started'):
    # Compatability with ImageClass_v<lt3> and ImageClass_v3
    if type(sequenceObj) is tuple:
      sequence,idx = hlp.convertObjOld(sequenceObj)
      sequenceObj = sequenceObj[0]
    else:
      sequence, idx = hlp.convertObj(sequenceObj)

    if update:
        settings = hlp.updateSettings(  settings,
                                        referencePeriod=float(sequenceObj[idx[0]].frameKeyPath('sync_info.reference_period')),
                                        referenceFrame=sequenceObj[idx[0]].frameKeyPath('sync_info.sync_settings.reference_frame'),
                                        barrierFrame=sequenceObj[idx[0]].frameKeyPath('sync_info.sync_settings.barrier_frame'))#also updates referenceFrameCount and targetSyncPhase
        if log:
            print('Reference sequence covers a period of {0} with {1} frames (including padding)'.format(settings['referencePeriod'],settings['referenceFrameCount']))
            print('Predicting for sync phase of {0}'.format(settings['targetSyncPhase']))
            print('Using barrier frame of {0}'.format(settings['barrierFrame']))

    #set drift-start
    drift = sequenceObj[idx[0]].frameKeyPath('drift_offset')[1:-1].split(', ')
    settings = hlp.updateSettings(settings,drift=[int(drift[1]),int(drift[0])])
    if log:
        print('Setting starting drift to {0}'.format(settings['drift']))

    #numFramesToUse array
    settings = rts.deduceBarrierFrameArray(settings)
    # pprint(settings)

    #set-up
    pastPhases = np.zeros([min(len(sequence),settings['maxReceivedFramesToStore']),3])#timestamp,phase,lowestSADframe
    triggers  = []#only those sent
    times = []#all
    phases = []#all
    bpm = []#beats per minute#all
    arrhythmia = []#residual on a perfectly periodic sawtooth#all

    for i in range(len(sequence)):
        #print('Frame index: {0}'.format(i))
        if i>=settings['maxReceivedFramesToStore']:
            pastPhases[0:-1,:] = pastPhases[1:,:]
            j = -1
        else:
            j = i

        ### Phase
        pp,dd,settings = rts.compareFrame(sequence[i], referenceFrames, settings, False)#ref pos including padding
        pp = ((pp-settings['numExtraRefFrames'])/settings['referencePeriod'])*(2*math.pi)#convert phase to 2pi base
        phases.append(pp)

        ### Timestamp
        tt = sequenceObj[idx[i]].frameKeyPath(timestamp)
        times.append(tt)
        pastPhases[j,0] = tt

        ### Arrhythmia #1 - Kind of works, highlights truncations better than expansions
        if j!=0:
            t0 = pastPhases[0,0]
            t1 = pastPhases[j-1,0]
            p1 = pp_old
            expected = p1 + ((t1-t0)/(settings['referencePeriod']/settings['framerate']) - (t1-t0)//(settings['referencePeriod']/settings['framerate'])) - ((tt-t0)/(settings['referencePeriod']/settings['framerate']) - (tt-t0)//(settings['referencePeriod']/settings['framerate']))
            residual = pp-expected
            # fix for wrapping?
            if residual > math.pi:
                residual = residual-2*math.pi
            elif residual < -math.pi:
                residual = residual+2*math.pi
            residual = abs(residual)
        else:
            residual = 0

        ### Cumulative Phase
        if j!=0:#make phase cumulative
            wrapped = False
            deltaPhase = pp-pp_old
            while deltaPhase<-math.pi:#c.f. lines 1798-1801 in SyncAnalyzer.mm; essentially this allows for small drops in phase due to SAD errors
                wrapped = True
                deltaPhase += 2*math.pi
            pastPhases[j,1] = pastPhases[j-1,1]+deltaPhase
        else:
            pastPhases[j,1] = pp

        ### Lowest SAD
        pastPhases[j,2] = np.argmin(dd)

        if log:
            print('Residual: ',residual)
        arrhythmia.append(residual)

        ### Predict Trigger
        # Don't predict if not done a whole period
        if i>settings['referencePeriod']:
            try:
                tr = rts.predictTrigger(np.copy(pastPhases[:i+1,:]),settings,True,log=log,output='seconds')
            except:
                print('----------------------------------------')
                print(settings['referenceFrameCount'])
                print(pastPhases[i,:])
                #pprint(settings)
                print('----------------------------------------')
                tr = 0
            if tr>0:
                tr,send,settings = rts.gotNewSyncEstimateTimeDelay(tt,tr,settings,log=log)
            else:
                send=-1
            if send>0:
                if log:
                    print('Trigger type {0} at {1}\tSENT'.format(send,tt+tr))
                triggers.append(tt+tr)

        ### Beats Per Minute
        framesToUse = min(60*settings['framerate'],pastPhases[:i+1,:].shape[0])
        period = (framesToUse/settings['framerate'])/60
        if i>settings['referencePeriod']:
            beats = (pastPhases[j,1]-pastPhases[j+1-framesToUse,1])/(2*math.pi)
        else:
            beats = np.nan
        bpm.append(beats/period)
        if log:
            print('Beats per minute: {0}; Beats per second: {1};'.format(bpm[-1],bpm[-1]/60))

        pp_old = float(pp)

    # plt.plot(times,bpm)
    # plt.plot(times,arrhythmia)
    # plt.plot(times,phases)
    # N = int(settings['referencePeriod']//2)
    # plt.plot(times[N//2:-N//2+1],np.convolve(arrhythmia, np.ones((N,))/N, mode='valid'),c='k')
    # plt.show()
    return triggers, settings, times, phases, bpm, arrhythmia

def loadReference(referenceNameFull,timestamp='time_processing_started',log=True):
    referenceObj, dummy = jil.LoadAllImages(referenceNameFull,True,1,0,-1,None,log=log)
    time = referenceObj[0].frameKeyPath(timestamp)
    referenceFrames,idx = hlp.convertObj(referenceObj)

    if os.path.isfile(os.path.join(referenceNameFull,'period.txt')):
        with open(os.path.join(referenceNameFull,'period.txt')) as f:
            tmp = [x for x in next(f).split()]
            period = float(tmp[0])
            if len(tmp)==4:
                drift = [int(x) for x in tmp[1:3]][::-1]
                target = float(tmp[-1])
            elif len(tmp)==3:
                drift = [int(x) for x in tmp[1:3]][::-1]
                target = None
            else:
                drift = None
                target = None
    else:
        period = None
        drift = None
        target = None

    return referenceFrames, (period, drift, target, time)

def getReference(referenceNameForm,stackNumber=None,formt='ref-frames'):
    if stackNumber is None:
        folder = referenceNameForm
    else:
        folder = referenceNameForm.format('Stack {0:04d}/'.format(stackNumber))
    subdirectories = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

    for sub in subdirectories:
        if sub.startswith(formt):
            referenceNameFull = os.path.join(folder,sub)

    return referenceNameFull

def phaseDiscrepancy(triggerTimes,times,phases,log=False):
    lb = min(phases)
    ub = max(phases)
    interpolatedPhases = []
    for t in range(len(triggerTimes)):
        timereceived = triggerTimes[t]
        if times[-1]<timereceived:
            if log:
                print('Out of time period... on trigger {0} of {1}... ignoring.'.format(t+1,len(triggerTimes)))
            continue
        # identify brightfield frame before and after fluorescence frame
        for p in range(len(times)):
            if times[p]>=timereceived:
                after = p
                afterTime = times[p]
                afterPhase = phases[p]
                before = p-1
                beforeTime = times[p-1]

                beforePhase = phases[p-1]
                break
        #inteprolate phase
        interpPos = (timereceived-beforeTime)/(afterTime-beforeTime)
        if afterPhase<(lb + (0.1*(ub-lb))) and beforePhase>(ub - (0.1*(ub-lb))):
            afterPhase = 1+afterPhase
        interpolatedPhases.append(((interpPos*(afterPhase-beforePhase))+beforePhase)%(2*math.pi))

    return interpolatedPhases

if __name__== "__main__":
    settings = hlp.initialiseSettings(drift=[-5,-2],
                        predictionLatency=0.01,
                        referencePeriod=42.41015632585688,framerate=80)
    settingsUpdated = deepcopy(settings)

    # load static references frames
    referenceNameFormat = '../notebooks/localdata/2017-02-09 18.35.13 vid overnight/{0}'
    referenceFrames = loadReference(getReference(referenceNameFormat,3))

    # set-up sequences
    sequenceName = '../notebooks/localdata/2017-02-09 18.35.13 vid overnight/Stack {0:04d}/Brightfield - Prosilica/'
    startingStack = 4
    endingStack = 10
    mT0 = []
    sT0 = []
    mT = []
    sT = []
    mF = []
    sF = []
    for stackNumber in range(startingStack,endingStack+1):
        print('Running for stack {0:04d}'.format(stackNumber))
        sequenceObj = jil.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)

        # Non-updated reference and settings
        if stackNumber==startingStack:#set initial settings
            print('First run, updating settings...')
            tr, settings, tt, pp0, bpm, arrhythmia = emulateSequence(sequenceObj,referenceFrames,settings,True)
        else:
            tr, settings, tt, pp0, bpm, arrhythmia = emulateSequence(sequenceObj,referenceFrames,settings,False)
        print('I counted a total of {0} triggers'.format(len(tr)))
        interpolatedPhase = phaseDiscrepancy(tr,tt,pp0)
        mu,sigma = norm.fit(np.asarray(interpolatedPhase)-settings['targetSyncPhase'])
        mF.append(mu)
        sF.append(sigma)

        # Updated reference and settings
        if stackNumber%3==1:#update reference frames every three stacks for non-static trial
            print('Updating reference frames...')
            referenceFramesUpdated = loadReference(getReference(referenceNameFormat,stackNumber-1))#won't need to be minus one in future

        tr, settingsUpdated, tt, pp, bpm, arrhythmia = emulateSequence(sequenceObj,referenceFramesUpdated,settingsUpdated,True,False)
        print('[updated] I counted a total of {0} triggers'.format(len(tr)))
        #using unupdated phases and target for fair comparison
        interpolatedPhase = phaseDiscrepancy(tr,tt,pp0)
        mu,sigma = norm.fit(np.asarray(interpolatedPhase)-settings['targetSyncPhase'])
        mT0.append(mu)
        sT0.append(sigma)
        #using updated phases and target for fair comparison
        interpolatedPhase = phaseDiscrepancy(tr,tt,pp)
        mu,sigma = norm.fit(np.asarray(interpolatedPhase)-settingsUpdated['targetSyncPhase'])
        mT.append(mu)
        sT.append(sigma)

    mT = np.asarray(mT)
    sT = np.asarray(sT)
    mF = np.asarray(mF)
    sF = np.asarray(sF)

    f1 = plt.figure()
    a1 = f1.add_axes([0,0,1,1])
    a1.plot(range(startingStack,endingStack+1),mT,c='g',label='Updated')
    a1.fill_between(range(startingStack,endingStack+1),mT-sT,mT+sT,facecolor='g',alpha=0.5)
    a1.plot(range(startingStack,endingStack+1),mF,c='r',label='Not')
    a1.fill_between(range(startingStack,endingStack+1),mF-sF,mF+sF,facecolor='r',alpha=0.5)
    a1.legend(loc='upper right')
    a1.set_title('Trigger Phase Discrepancy with and without Updates')
    a1.set_xlabel('Reference Stack ID')
    a1.set_xlim([1,endingStack])
    a1.set_ylabel('Trigger Phase Discrepancy')
    #a1.set_ylim([-0.1,0.3])
    a1.set_ylim([-0.5,0.75])
    #a1.set_ylim([-0.6,0.6])
    plt.show()

    print(mT,sT)
    print(mF,sF)
