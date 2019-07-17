from analyze_dataset import *

basePath = '/Users/jonny/Movies/2016-06-30 17.46.41 vid piv flow heart plane'

# It is helpful to crop tighter, since this is a particularly large dataset
crop = (0, 230, 60, 210)

if True:
	# Full dataset
    AnalyzeDataset(basePath, \
                   imageRangeList = [(64282, 107078)], \
                   annotationList = ['Brightfield - Prosilica', 'Red - QIClick Q35979'], \
                   periodRange = np.arange(45, 55, 0.1), \
                   downsampling = 2, \
                   numSamplesPerPeriod = 60, \
                   cropRect = crop)
else:
    # Quick subset test
    AnalyzeDataset(basePath, [(64282, 64882)], 'Red - QIClick Q35979', periodRange = np.arange(20, 30, 0.1), numSamplesPerPeriod = 50, cropRect=crop)
