from analyze_dataset import *

basePath = '/Users/jonny/Movies/sync test datasets/2016-06-28 13.01.05 vid continuous fluor + bf'
if True:
	frameRanges = [(43740, 52676)]
else:
	frameRanges = [(43740, 45739)]

# Note that I restrict the period range to avoid getting caught out by
# major twitching of the fish around frame 1200
# (and there is also something more minor around frame 1500)
# Also fewer samples per period in order to avoid running out of memory on 32-bit python
AnalyzeDataset(basePath, frameRanges, 'Green - QIClick Q35977', periodRange = np.arange(23, 32, 0.1), numSamplesPerPeriod = 50)
