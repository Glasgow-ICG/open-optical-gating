from analyze_dataset import *

basePath = '/Users/spim/jonny/gata 30.6.16/2016-06-30 18.26.31 vid stepwise vessel piv'
if True:
	frameRanges = [(308188, 431002)]
elif True:
    # Subset for testing of sync quality
    frameRanges = [(308188, 312999)]
    AnalyzeDataset(basePath, frameRanges, ['Brightfield subset'], periodRange = np.arange(45, 55, 0.1), numSamplesPerPeriod = 50, source='Brightfield subset')
    exit(0)
else:
    # Very short subset just for basic testing that code runs
    frameRanges = [(308188, 309188)]

AnalyzeDataset(basePath, frameRanges, ['Red - QIClick Q35979', 'Brightfield - Prosilica'], periodRange = np.arange(45, 55, 0.1), numSamplesPerPeriod = 60)
