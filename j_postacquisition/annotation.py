from phase_wrap_interpolation import *
import plist_wrapper as jPlist
import numpy as np
from tqdm import *
from image_loading import *

def InvalidPhase(fluorPhases):
    return fluorPhases > 6.9

def InterpolateForImagePhases(fluorImages, knownTimes, knownPhases, knownRefs, smoothFluorTimestamps=False):
    # Interpolate to get fluor phases, using the known information from the brightfield data
    fluorTimes = []
    for i in range(len(fluorImages)):
        fluorTimes.append(fluorImages[i].timestamp)
    fluorTimes = np.array(fluorTimes)

    # Optional: smooth the fluorescence timestamps, on the assumption that the camera was running at a constant framerate
    if smoothFluorTimestamps:
        index = np.arange(len(fluorTimes))
        fit = np.polyfit(index, fluorTimes, 1)
        fit_fn = np.poly1d(fit)
        fluorTimes = fit_fn(index)

    # Now interpolate to obtain the phases
    fluorPhases = interpolate_with_phase_wrap(fluorTimes, knownTimes, knownPhases)
    fluorRefs = interpolate_with_phase_wrap(fluorTimes, knownTimes, knownRefs)

    # Try setting all nans to phase 7,  that may help with sorting etc
    for i in range(len(fluorPhases)):
        if (np.isnan(fluorPhases[i])):
            fluorPhases[i] = 7
    
    for i in range(len(fluorImages)):
        fluorImages[i].phase = fluorPhases[i]
        fluorImages[i].refFrame = fluorRefs[i]

    sortedFluorImages = sorted(fluorImages, key=operator.attrgetter('phase'))
    print(len(np.where(InvalidPhase(fluorPhases))[0]), '/', len(fluorPhases), 'failed to recover a phase')

    return sortedFluorImages


def EditOriginalPlists(fluorPath, fluorImages, key):
    # Edit the original plist to add the phase information
    # Note that this will lead to some textual changes to the content of the plist -
    # specifically, 'real' types stored as e.g. '2' will be changed to '2.0'.
    # That should not have any effect on any code reading the plist.

    # All plist caches will be invalid once we have edited the plists!
    # Best thing we can do at this point is just to delete any .npy files in the fluor directory
    DeleteAnyCachesInDir(fluorPath)

    # Now do the actual editing
    for i in tqdm(range(len(fluorImages)), desc='edit plists'):
        if ((fluorImages[i].phase != 7) and (np.isnan(fluorImages[i].phase) != True)):
            # We have a valid phase for this frame
            fluorImages[i]._frameMetadata[key] = fluorImages[i].phase
            fluorImages[i]._frameMetadata['ref_from_offline_sync_analysis'] = fluorImages[i].refFrame
        elif key in fluorImages[i]._frameMetadata:
            # We do not have a valid phase for this frame. We should actively
            # remove any key that is already present from any previous analysis attempt
            del fluorImages[i]._frameMetadata[key]
            del fluorImages[i]._frameMetadata['ref_from_offline_sync_analysis']
    ResaveMetadataAfterEditing(fluorImages)


def AnnotateFluorChannel(fluorPath, knownTimes, knownPhases, knownRefs, earlyTruncation=-1, keyToWriteForAnnotation='postacquisition_phase', fluorTimestampKey='time_exposed', smoothFluorTimestamps=False):
    print('Define phase (key %s) for %s' % (keyToWriteForAnnotation, fluorPath))
    (fluorImages, p) = LoadAllImages(fluorPath, loadImageData=False, earlyTruncation=earlyTruncation, timestampKey=fluorTimestampKey)
    InterpolateForImagePhases(fluorImages, knownTimes, knownPhases, knownRefs, smoothFluorTimestamps=smoothFluorTimestamps)
    EditOriginalPlists(fluorPath, fluorImages, keyToWriteForAnnotation)
    # Although the caller will probably not normally care about this, we provide the
    # images array in case it is of use
    return fluorImages
