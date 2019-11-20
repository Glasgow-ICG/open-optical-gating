import math
import numpy as np
from pprint import pprint

def convertObj(sequenceObj,timestamp='time_processing_started', sortByTimestamp=True):
    # Convert to 3D numpy array (ZXY), assumes correctly sorted
    # Works for new image_class v3 (multipage tiffs)
    # Should work for old formats
    if type(sequenceObj) is tuple:
        sequenceObj = sequenceObj[0]
    if type(sequenceObj) is list:
        sequenceObj = np.array(sequenceObj)
    sequenceLength = sequenceObj.shape[0]
    sz = sequenceObj[0].image.shape
    sequence = np.zeros([sequenceLength,sz[0],sz[1]],sequenceObj[0].image.dtype)
    times = np.zeros(sequenceLength)
    for i in range(sequenceLength):
        sequence[i] = np.copy(sequenceObj[i].image)
        times[i] = sequenceObj[i].frameKeyPath(timestamp)
    if sortByTimestamp:
		# Sorting is optional - for JT's purposes he wants to save strictly in the order provided in sequenceObj
        idx = np.argsort(times)
        sequence = sequence[idx,:,:]

    return sequence, idx

def convertObjOld(sequenceObj,timestamp='time_processing_started'):
    # Sort sequence (just in case) and convert to list of numpy arrays
    # works for old format (multiple files)
    sequenceObj = sequenceObj[0]
    sequenceLength = sequenceObj.size
    sz = sequenceObj[0].image.shape
    sequence = np.zeros([sequenceLength,sz[0],sz[1]],sequenceObj[0].image.dtype)
    times = np.zeros(sequenceLength)
    for i in range(sequenceLength):
        sequence[i] = np.copy(sequenceObj[i].image)
        times[i] = sequenceObj[i].frameKeyPath(timestamp)
    idx = np.argsort(times)
    sequence = sequence[idx,:,:]

    return sequence, idx

def initialiseSettings(drift=[0,0],framerate=80,referencePeriod=0.0,
                       barrierFrame=0.0,extrapolationFactor=1.5,
                       maxReceivedFramesToStore=260,maxFramesForFit=32,minFramesForFit=3,
                       predictionLatency=15,referenceFrame=0.0,numExtraRefFrames=2):
    settings = {}
    settings.update({'drift':drift})#starting drift corrections
    settings.update({'framerate':framerate})#starting est frame rate
    settings.update({'referencePeriod':referencePeriod})#reference period in frames
    if barrierFrame > 0.0:
        settings.update({'barrierFrame':((barrierFrame-numExtraRefFrames)%referencePeriod)+numExtraRefFrames})#barrier frame in frames
    else:
        settings.update({'barrierFrame':barrierFrame})
    settings.update({'extrapolationFactor':extrapolationFactor})
    settings.update({'maxReceivedFramesToStore':maxReceivedFramesToStore})#maximum number of frames to stores, used to prevent memory filling
    settings.update({'maxFramesForFit':maxFramesForFit})#frames to fit for prediction (max)
    settings.update({'minFramesForFit':minFramesForFit})#frames to fit for prediction (min)
    settings.update({'predictionLatency':predictionLatency})#prediction latency in milliseconds
    if referenceFrame > 0.0:
        settings.update({'referenceFrame':referenceFrame%referencePeriod})#target phase in frames
    else:
        settings.update({'referenceFrame':referenceFrame})
    settings.update({'numExtraRefFrames':numExtraRefFrames})#padding number

    # automatically added keys
    settings.update({'referenceFrameCount':math.ceil(settings['referencePeriod'])+(2*settings['numExtraRefFrames'])})#number of reference frames including padding
    if referencePeriod > 0.0:
      settings.update({'targetSyncPhase':2*math.pi*(settings['referenceFrame']/settings['referencePeriod'])})#target phase in rads
    else:
      settings.update({'targetSyncPhase':0})#target phase in rads
    settings.update({'lastSent':0.0})
    #settings.update({'frameToUseArray':[0]})#this should be created locally when needed


    return settings

def updateSettings(settings,drift=None,framerate=None,referencePeriod=None,
                       barrierFrame=None,extrapolationFactor=None,
                       maxReceivedFramesToStore=None,maxFramesForFit=None,minFramesForFit=None,
                       predictionLatency=None,referenceFrame=None,numExtraRefFrames=None):

    if drift is not None:
        settings['drift'] = drift
    if framerate is not None:
        settings['framerate'] = framerate
    if referencePeriod is not None:
        settings['referencePeriod'] = referencePeriod
    if numExtraRefFrames is not None:
        settings['numExtraRefFrames'] = numExtraRefFrames
    if barrierFrame is not None:
        settings['barrierFrame'] = ((barrierFrame-settings['numExtraRefFrames'])%settings['referencePeriod'])+settings['numExtraRefFrames']
    if extrapolationFactor is not None:
        settings['extrapolationFactor'] = extrapolationFactor
    if maxReceivedFramesToStore is not None:
        settings['maxReceivedFramesToStore'] = maxReceivedFramesToStore
    if maxFramesForFit is not None:
        settings['maxFramesForFit'] = maxFramesForFit
    if minFramesForFit is not None:
        settings['minFramesForFit'] = minFramesForFit
    if predictionLatency is not None:
        settings['predictionLatency'] = predictionLatency
    if referenceFrame is not None:
        settings['referenceFrame'] = referenceFrame%settings['referencePeriod']

    # automatically added keys
    settings.update({'referenceFrameCount':math.ceil(settings['referencePeriod'])+(2*settings['numExtraRefFrames'])})#number of reference frames including padding
    settings.update({'targetSyncPhase':2*math.pi*(settings['referenceFrame']/settings['referencePeriod'])})#target phase in rads

    return settings
