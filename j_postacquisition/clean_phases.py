# Strips out any phase keys that may have been added to the plists
# Gives us a clean slate for a subsequent analysis.

from analyze_dataset import *

def StripFrameKeys(plistPath, keyList):
    # Read information from the plist
    pl = jPlist.readPlist(plistPath)

    if ((not ('metadata_version' in pl)) or (pl['metadata_version'] == 1)):
        # Delete the keys
        for k in keyList:
            if k in pl:
                del pl[k]
    else:
        # Delete the keys
        for f in pl['frames']:
            for k in keyList:
                if k in f:
                    del f[k]
    
    # Save the plist
    jPlist.writePlist(pl, plistPath)

def CleanPhasesInDir(dirPath):
    fileList = []
    for file in tqdm(os.listdir(dirPath), desc='cleaning phases'):
        if file.endswith(".plist"):
            StripFrameKeys(dirPath+'/'+file, ['sync_info.phase', 'postacquisition_phase', 'phase_from_offline_sync_analysis'])


basePath = '/Users/jonny/Movies/2016-06-09 16.37.35 vid slow heart belly scan'
CleanPhasesInDir(basePath+'/Brightfield - Prosilica')
CleanPhasesInDir(basePath+'/Red - QIClick Q35979')

