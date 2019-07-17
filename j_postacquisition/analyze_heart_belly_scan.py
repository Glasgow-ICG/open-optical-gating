from analyze_dataset import *

basePath = '/Users/jonny/Movies/2016-06-09 16.37.35 vid slow heart belly scan'

# Dataset needs special handling due to the frame number reset from 65535 to 1
# Dont read the final bit where the focus correction goes crazy
AnalyzeDataset(basePath, [(54863, 65535), (1, 12000)], ['Red - QIClick Q35979'])
