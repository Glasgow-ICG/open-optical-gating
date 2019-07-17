from shifts import *
from periods import *
from image_loading import *
from drift_correction import *
from shifts_global_solution import *
from tqdm import *
import importlib
import matplotlib.image as img
import plist_wrapper as jPlist
from phase_assignment import *
from annotation import *
import sys

def AnnotateDatasetUsingBrightfieldPhases(basePath, imageRangeList=None, fluorFolders=[], source='Brightfield - Prosilica', key='phase_from_offline_sync_analysis', \
                                          keyToWrite=None, fluorTimestampKey='time_exposed', smoothPhaseFactor=0, smoothFluorTimestamps=False):
    
    # This code uses phases already entered in the brightfield plist
    # (either using postacquisition code, or by running a special build of my realtime sync),
    # and interpolates to fill in equivalent timestamps for the fluor channel.
    # The realtime sync may not give the absolute best phase calculations, but it's a quick way of getting *something*.
    if keyToWrite is None:
        keyToWrite = key        # May want this to be different, e.g. if key='sync_info.phase'
    pathFormat = '%s/'+source+'/%06d.tif'
    if (imageRangeList == None):
        (images, dummy) = LoadAllImages('%s/%s' % (basePath, source), loadImageData = False)
    else:
        assert(len(imageRangeList) > 0)
        (firstImage, lastImage) = imageRangeList[0]
        (images, _) = LoadImages(basePath, pathFormat, firstImage, lastImage - firstImage + 1, loadImageData=False)

    print('First image', os.path.basename(images[0]._path))
    print('Last image', os.path.basename(images[-1]._path))
    knownTimes = []
    knownPhases = []
    knownRefs = []
    for i in range(len(images)):
        try:
            ph = images[i].frameKeyPath(key) % (2*np.pi)   # Modulo is needed if key='sync_info.phase'
            t = images[i].frameKeyPath('time_processing_started')
            knownPhases.append(ph)
            knownTimes.append(t)
            try:
                r = images[i].frameKeyPath('ref_from_offline_sync_analysis')
            except:
                r = -1
            knownRefs.append(r)
        except:
            pass
    knownTimes = np.array(knownTimes)
    knownPhases = np.array(knownPhases)
    knownRefs = np.array(knownRefs)
    
    # Apply smoothing to the phases if desired, handling phase-wrap appropriately
    smoothed = 0
    for delta in range(-smoothPhaseFactor, smoothPhaseFactor+1):
        neighbour = np.roll(knownPhases, delta)      # This behaves wrong at each end of array but we will handle that later
        # Handle phase wrap
        wrap = np.where((knownPhases > 1.5*np.pi) & (neighbour < 0.5*np.pi))
        neighbour[wrap] = neighbour[wrap] + 2.0*np.pi
        wrap = np.where((neighbour > 1.5*np.pi) & (knownPhases < 0.5*np.pi))
        neighbour[wrap] = neighbour[wrap] - 2.0*np.pi
        # Update running sum
        smoothed = smoothed + neighbour
    # Turn running sum into average
    smoothed = smoothed / (smoothPhaseFactor*2+1)
    # Handle additional phase-wraps that might have been introduced in the averaging
    smoothed = smoothed % (2*np.pi)
    # Update knownPhases
    unsmoothed = knownPhases.copy()
    knownPhases = smoothed
    # Snip off the end elements where we didn't have enough values to do the average properly
    if (smoothPhaseFactor > 0):
        knownPhases = knownPhases[smoothPhaseFactor:-(smoothPhaseFactor)]
        knownTimes = knownTimes[smoothPhaseFactor:-(smoothPhaseFactor)]
        unsmoothed = unsmoothed[smoothPhaseFactor:-(smoothPhaseFactor)]
    
    # Do the actual annotation
    for fluorFolder in fluorFolders:
        fluorImages = AnnotateFluorChannel('%s/%s' % (basePath, fluorFolder), knownTimes, knownPhases, knownRefs, keyToWriteForAnnotation='phase_from_offline_sync_analysis', fluorTimestampKey=fluorTimestampKey, smoothFluorTimestamps=smoothFluorTimestamps)
    return (knownTimes, knownPhases, unsmoothed, knownRefs)


if __name__ == "__main__":
    assert(len(sys.argv) == 4)
    AnnotateDatasetUsingBrightfieldPhases(sys.argv[1], fluorFolders=[sys.argv[2]], source=sys.argv[3])
