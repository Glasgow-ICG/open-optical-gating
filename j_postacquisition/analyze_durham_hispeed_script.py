from analyze_dataset import *

basePath = '/Volumes/Jonny_Backup_Physics/spim_vids/2013-11-28 Durham high speed brightfield'
frameRanges = None
#GenerateFakePlists(basePath+'/fish1bf', framerate = 800, stemLength = 8, fakeZScan=True)
AnalyzeDataset(basePath, frameRanges, ['fish1bf'], periodRange = np.arange(750, 790,), numSamplesPerPeriod = 1000, alpha = 50, source='fish1bf', sourceTimestampKey = 'time_processing_started', fluorTimestampKey = 'time_processing_started')
