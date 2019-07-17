from analyze_dataset import *

basePath = '/Users/jonny/Movies/sync test datasets/2016-06-28 13.01.05 vid continuous fluor copy'
if True:
	frameRanges = [(66166, 68399)]
else:
	frameRanges = [(66166, 68165)]

AnalyzeDataset(basePath, firstImage, lastImage, 'Green - QIClick Q35977', source='Green - QIClick Q35977', numSamplesPerPeriod=20, periodRange=np.arange(3,7,0.05), alpha=2)
