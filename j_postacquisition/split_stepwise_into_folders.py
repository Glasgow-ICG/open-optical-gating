from image_loading import *
from tqdm import *
import os

def SplitDatasetIntoFolders(basePath, destPath=None, source = 'Red - QIClick Q35979'):
    # This code sorts through the files and looks for changes to the 'stepwise_definitely_stationary' property.
    # Batches of files where the stages are stationary are split out into separate folders
    if (destPath is None):
        destPath = basePath
    prevWasStationary = False
    folderToUse = None
    counter = 0

    # This code is probably not going to work with the new plist format - it will need some thought and further work to update it. 
    assert(False)
    
    (images, dummy) = LoadAllImages(basePath+'/'+source, loadImageData=False)
    for im in tqdm(images, desc='Copying images'):
        stat = im.z_scan()['stepwise_definitely_stationary']
        if (stat):
            if (stat == prevWasStationary):
                # We will use the same folder as the last image
                assert(folderToUse is not None)
            else:
                # Create a new folder
                folderToUse = destPath+('/%03d' % counter)
                os.system('mkdir -p "%s"' % folderToUse)
                counter = counter + 1
            # Copy the image (and plist) to the folder
            os.system('cp "%s" "%s"' % (im._path, folderToUse))
            os.system('cp "%s" "%s"' % (im._plistPath, folderToUse))
        prevWasStationary = stat


if __name__ == "__main__":
    SplitDatasetIntoFolders('/Users/jonny/Movies/temp', '/Users/jonny/Movies/temp/split', source = 'Brightfield subset')
