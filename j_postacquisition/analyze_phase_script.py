from analyze_dataset import *

basePath = '/Users/jonny/Movies/2017-01-11 PARTIAL/2017-01-11 14.54.34 vid tail piv'
frameRanges = None
if False:
    # Full sync of both channels
    AnalyzeDataset(basePath, frameRanges, ['Red - QIClick Q35979', 'Green - QIClick Q35977'], periodRange = np.arange(22, 32, 0.1), numSamplesPerPeriod = 60)
else:
    # Subset for testing of sync quality
    AnalyzeDataset(basePath, [(64002, 74002)], ['Brightfield subset'], periodRange = np.arange(10, 80, 0.2), numSamplesPerPeriod = 60, alpha=20)
#    AnalyzeDataset(basePath, frameRanges, ['Brightfield subset'], periodRange = np.arange(10, 80, 0.2), numSamplesPerPeriod = 60, source='Brightfield subset', alpha=20, plotAllPeriods=True)

