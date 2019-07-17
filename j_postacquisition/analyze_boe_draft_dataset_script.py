from analyze_dataset import *

basePath = '/Volumes/Jonny Data Work/spim_vids_2/2017-01-11 gata+flk heart and vessels/2017-01-11 15.37.20 vid heart piv stack'
frameRanges = None
if True:
    # Full sync of both channels
    AnalyzeDataset(basePath, frameRanges, ['Red - QIClick Q35979', 'Green - QIClick Q35977'], periodRange = np.arange(40, 48, 0.1), numSamplesPerPeriod = 60)
else:
    # Subset for testing of sync quality
    AnalyzeDataset(basePath, frameRanges, ['Brightfield subset'], periodRange = np.arange(40, 48, 0.1), numSamplesPerPeriod = 60, source='Brightfield subset')

