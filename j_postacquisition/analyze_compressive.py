from analyze_dataset import *

basePath = '/Users/jonny/Movies/pigmAexcerpt'

# Use this to generate plists for the bare tiff files
#GenerateFakePlists(basePath)

# Demonstrating what happens if we read in Alex's 7-channel compressive data and analyze it
AnalyzeDataset(basePath, imageRangeList=None, annotationList=None, downsampling=1, numSamplesPerPeriod = 100, source = '.', periodRange = np.arange(60, 140, 1), plotAllPeriods=True, applyDriftCorrection=False)
