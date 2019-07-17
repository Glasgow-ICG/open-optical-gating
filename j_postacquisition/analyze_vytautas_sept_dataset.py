from analyze_dataset import *

basePath = '/Users/jonny/Movies/2016-09-19 16.09.55  as above bigger gap 20s_4um'
frameRanges = None
if True:
    # Full sync of both channels
    AnalyzeDataset(basePath, frameRanges, ['Red - QIClick Q35979', 'Green - QIClick Q35977'], periodRange = np.arange(22, 32, 0.1), numSamplesPerPeriod = 60)
else:
    # Subset for testing of sync quality
    AnalyzeDataset(basePath, frameRanges, ['Brightfield subset'], periodRange = np.arange(22, 32, 0.1), numSamplesPerPeriod = 60, source='Brightfield subset')

