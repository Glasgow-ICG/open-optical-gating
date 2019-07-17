from analyze_dataset import *

if False:
    basePath = '/Users/jonny/Movies/for_sync_vytautas_13_06_2016/2016-06-11 17.22.31 into heart piv 5msgap'
    periodRange = np.arange(34, 43, 0.1)
    rangeList = None
else:
    basePath = '/Users/jonny/Movies/for_sync_vytautas_13_06_2016/2016-06-11 12.12.59 piv10ms_zscan42um_10s'
    periodRange = np.arange(46, 57, 0.1)
    rangeList = [(12801, 13400)]
    rangeList = None

AnalyzeDataset(basePath, rangeList, ['Allied Vision Technologies GS650 0001f61c', 'QImaging Retiga 1350B'], periodRange = periodRange, numSamplesPerPeriod = 50, source='Allied Vision Technologies GS650 0001f61c')
