from analyze_dataset import *

frameRanges = None
basePath='/Users/jonny/Movies/Example_compr_sens_datasets/Jonny revisiting raw datasets'
if True:
    # Analyze photodiode channels, with 4-point averaging to smooth out 50Hz interference
    # Note that we look for time_processing_started in the photodiode channel. That is just a synthetic timestamp, and I presume that's the key I suggested Alex fills in.
    (kt1, kp1, _, _, _, _) = AnalyzeDataset(basePath, frameRanges, ['pd_darkfield'], periodRange=np.arange(100, 140, 0.2), numSamplesPerPeriod=200, source='pd_darkfield', applyDriftCorrection=False, downsampling=1, interpolationDistanceBetweenSequences=4, rollingAverage=True, sourceTimestampKey='time_processing_started', fluorTimestampKey='time_processing_started')
    # Also phase-stamp a copy of the widefield data, but using the phases calculated from the photodiode data
    # (code needs updating now I have switched datasets)
    # AnnotateFluorChannel('%s/%s' % (basePath, 'widefield_with_photodiode_phase_stamps'), kt1*5, kp1, -1, fluorTimestampKey='synthetic_timestamp')

if True:
    (kt2, kp2, _, _, _, _) = AnalyzeDataset(basePath, frameRanges, ['pd_brightfield'], periodRange=np.arange(100, 140, 0.2), numSamplesPerPeriod=200, source='pd_brightfield', alpha=100, applyDriftCorrection=False, downsampling=1, interpolationDistanceBetweenSequences=4, rollingAverage=True, sourceTimestampKey='time_processing_started', fluorTimestampKey='time_processing_started')
    # (code needs updating now I have switched datasets)
    # AnnotateFluorChannel('%s/%s' % (basePath, 'widefield_with_photodiode_phase_stamps'), kt2*5, kp2, -1, fluorTimestampKey='synthetic_timestamp')

if True:
    (kt3, kp3, _, _, _, _) = AnalyzeDataset(basePath, frameRanges, ['wf_darkfield'], periodRange=np.arange(30, 56, 0.1), numSamplesPerPeriod=80, source='wf_darkfield', applyDriftCorrection=False, downsampling=1, interpolationDistanceBetweenSequences=4, sourceTimestampKey='timestamp', fluorTimestampKey='timestamp')

if True:
    (kt4, kp4, _, _, _, _) = AnalyzeDataset(basePath, frameRanges, ['wf_brightfield'], periodRange=np.arange(30, 56, 0.1), numSamplesPerPeriod=80, source='wf_brightfield', applyDriftCorrection=False, downsampling=1, interpolationDistanceBetweenSequences=4, sourceTimestampKey='timestamp', fluorTimestampKey='timestamp')



def diffs(knownTimes, knownPhases):
    y = knownPhases[1:] - knownPhases[0:knownPhases.size-1]
    y[np.where(y<0)]=np.nan
    x = knownTimes - knownTimes[0]
    x = x[1:]
    return (x,y)

(x1,y1) = diffs(kt1, kp1)
x1 = x1 * 2.5
y1 = y1 * 2.5
if True:
    (x2,y2) = diffs(kt2, kp2)
    x2 = x2 * 2.5
    y2 = y2 * 2.5
    (x3,y3) = diffs(kt3, kp3)
    (x4,y4) = diffs(kt4, kp4)
else:
    x2=x1
    x3=x1
    x4=x1
    y2=y1
    y3=y1
    y4=y1

plt.gcf().clear()
plt.ylim((0,0.5))
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.plot(x4,y4)
plt.plot(np.array([0,7]), np.array([0.07853982, 0.07853982]))
plt.show()