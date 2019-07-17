from annotate_dataset_em_cam import *

basePath = '/Users/jonny/Movies/sync test datasets/2016-06-28 13.01.05 vid continuous fluor + bf'
if True:
	frameRanges = [(43740, 52676)]
else:
	# Reduced range for test purposes
	frameRanges = [(43740, 45739)]

AnnotateDatasetUsingEmulatedCameraPhases(basePath, frameRanges, fluorFolder1 = 'Green - QIClick Q35977', key='phase_from_offline_sync_analysis')
