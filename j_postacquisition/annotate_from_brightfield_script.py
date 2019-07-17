from annotate_dataset_using_brightfield import *

def FixPlists(path):
    # Fixes plists garbled by previous bug in plist annotation.
    # Hopefully this code will never be needed again, but I am relucant to delete it in case there turns out to be a stack around somewhere that is still affected!
    # IMPORTANT: must not be run on a "correct" (or fixed) folder, or it will garble it again!
    fileList = []
    for file in os.listdir(path):
        if (file.endswith(".plist")):
            fileList.append(path+'/'+file)
    plists = []
    for plistPath in tqdm(fileList, desc='loading plists'):
        plists.append(jPlist.readPlist(plistPath))
    
    for i in tqdm(range(len(plists)-1), desc='fixing plists'):
        # Each plist should keep its final one, and take all but the final one from the next set
        thisNewPlist = plists[i]
        thisNewPlist['frames'] = [thisNewPlist['frames'][-1]]
        thisNewPlist['frames'].extend(plists[i+1]['frames'][:-1])
        jPlist.writePlist(thisNewPlist, fileList[i])

basePath = '/Users/jtlab/jonny/2018-04-27 12.48.12 vid liebling'
AnnotateDatasetUsingBrightfieldPhases(basePath, None, ['Retiga 1350B'], source='Allied Vision Technologies GS650 0001f61c', key='phase_from_offline_sync_analysis')
