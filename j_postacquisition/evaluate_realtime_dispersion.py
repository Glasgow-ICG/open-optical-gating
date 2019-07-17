# This code compares the observed phases for received fluorescence frames against the target sync phase.
# HOWEVER, all this analysis should currently be taken with a big pinch of salt.
# This is mainly because the exposed/received timestamps on fluorescence frames are not accurate enough,
# and much of the observed apparent dispersion in phase is actually due to these timestamp errors.
# Note though that the Ximea camera provides much more accurate timestamps using its own internal timebase.
# This code should be improved by getting fluorescence timestamps from the Ximea camera timebase and relating that
# to the Prosilica camera timebase, and using that for more accurate phase-stamping.
# Alternatively, it is possible to parse the sync log files and simply look at the interpolated phases for the
# (Prosilica) times at which triggers were sent - without any direct reference to the received fluorescence frames
# that were acquired in response to those triggers.

from image_loading import *
from tqdm import *
import os
from matplotlib import pyplot as plt

def EvaluateRealtimeDispersion(basePath, bfSource = 'Brightfield - Prosilica', fluorSource = 'Green - QIClick Q35977'):
    # For each of the (synchronized) fluorescence files, find the accompanying brightfield files.
    # Examine their metadata and interpolate to estimate the phase at which the fluorescence image was acquired
    
    (brightfield, dummy) = LoadAllImages(basePath+'/'+bfSource, loadImageData=False)#, earlyTruncation=1000)
    (fluor, dummy) = LoadAllImages(basePath+'/'+fluorSource, loadImageData=False)#, earlyTruncation=20)
    
    # Build up map of timestamp vs brightfield phase
    bfTimes = []
    bfPhases = []
    bfImages = []
    for im in tqdm(brightfield, desc='Tabulating phases'):
        # Note that I use there ref pos rather than the phase, since the phase is unwrapped which makes things slightly more fiddly to handle
        if 'sync_info' in im._frameMetadata:
            bfTimes.append(im.time_exposed())
            bfPhases.append(im.frameKeyPath('sync_info.ref_pos_without_padding'))        # ?? must decide what to do if this key is absent...
            bfImages.append(im)
    bfTimes = np.array(bfTimes)
    bfPhases = np.array(bfPhases)
    bfImages = np.array(bfImages)
    fluorTimes = []
    fluorImages = []
    for im in tqdm(fluor, desc='Tabulating phases'):
        # Skip a fluor frame that lies outside the brightfield range we have (unusual!)
        if (im.time_exposed() > bfTimes[0]) and (im.time_exposed() < bfTimes[-1]):
            # Make a note of the timestamp for this fluor frame
            fluorTimes.append(im.time_exposed())
            fluorImages.append(im)
    fluorTimes = np.array(fluorTimes)
    fluorImages = np.array(fluorImages)
    
    # Interpolate to determine phases
    afterIndices = np.searchsorted(bfTimes, fluorTimes)
    beforeIndices = afterIndices - 1
    fracs = (fluorTimes - bfTimes[beforeIndices]) / (bfTimes[afterIndices] - bfTimes[beforeIndices])
    phases = bfPhases[beforeIndices] * (1 - fracs) + bfPhases[afterIndices] * fracs
    
    return (fluorImages, bfImages[beforeIndices], phases)

def EvaluateStackFolder(path, histPath, bfSource = 'Brightfield - Prosilica', fluorSource = 'Green - QIClick Q35977'):
    (fluorImages, bfImageCounterparts, phases) = EvaluateRealtimeDispersion(path, bfSource, fluorSource)
    
    # Do an initial histogram analysis to estimate the target sync phase,
    # since unfortunately I didn't previously record the period in an easily-accessible way.
    (hist, bin_edges) = np.histogram(phases, bins=range(0, 100))
    estSyncPhase = bin_edges[np.argmax(hist)]
    if (len(bfImageCounterparts > 0)):
        targetSyncPhase = bfImageCounterparts[0].frameKeyPath('sync_info.sync_settings.reference_frame')
    else:
        # Who knows what the target phase was!
        # We have no frames though, so it doesn't really matter...
        targetSyncPhase = estSyncPhase
    
    if True:
        plt.clf()
        if True:
            # Do a new histogram centered on this value
            binRange = np.arange(targetSyncPhase-4, targetSyncPhase+6, 0.25)
            plt.title('target %.2lf' % targetSyncPhase)
            [n, b, _] = plt.hist(np.clip(phases, binRange[0], binRange[-1]), bins=binRange)
            plt.plot([targetSyncPhase, targetSyncPhase], [0, np.max(n)+5], 'r')
            plt.xlim([binRange[0], binRange[-1]])
        else:
            plt.hist(phases, 100)
        #plt.show()
        mkdir_p(histPath)
        plt.savefig('%s/hist %s' % (histPath, os.path.basename(path)))

    if (False):
        # Print the histogram data to the console
        for (h, b) in zip(hist, bin_edges):
            print (b, h)

    # Identify the indices of the frames whose phases are more than +-3 away from the most popular histogram bin
    # The [0] is because (for some reason...?) we get a tuple containing a single numpy array...
    centralBin = bin_edges[np.argmax(hist)]
    estOutliers = np.where((phases < (estSyncPhase-3.0)) | (phases > (estSyncPhase+3.0))) [0]
    targetOutliers = np.where((phases < (targetSyncPhase-3.0)) | (phases > (targetSyncPhase+3.0))) [0]
	
    # Annotate plists of outlier frames
    print ('%d or %d outliers from total %d for %s' % (len(estOutliers), len(targetOutliers), len(fluorImages), os.path.basename(path)))

    if False:
        #        for outlier in tqdm(outliers, desc='editing plists'):
        for i in tqdm(range(len(fluorImages)), desc='editing plists'):
            #print (fluorImages[outlier]._plistPath, phases[outlier])
            if True:
                fluorImages[i]._frameMetadata['estimated_ref_frame_error'] = abs(phases[i] - estSyncPhase)
                fluorImages[i]._frameMetadata['sync_settings_from_master'] = bfImageCounterparts[i].frameKeyPath('sync_info.sync_settings')
        ResaveMetadataAfterEditing(fluorImages)



if __name__ == "__main__":
    if True:
        for i in range(1, 245+1):
            EvaluateStackFolder('/Volumes/Tosh3TB/flkmCherry post 2dpf looping 6/2018-07-26 11.01.47 vid/Stack %04d' % i, 'hists', fluorSource='mcherry - QIClick Q35977')
    else:
        # Second dataset
        for i in range(1, 263+1):
            EvaluateStackFolder('/Users/spim/Finn/trab 7 myl7 mkatecaax h2bgfp/2018-07-30 13.32.41 vid/Stack %04d' % i, '/Users/spim/jonny/hists_trab7')
    #(fluorImages, phases) = EvaluateRealtimeDispersion('/Volumes/Jonny Data Work/spim_vids_2/2016-06-09 benchmark sync and postprocessable data/2016-06-09 16.50.18 vid heart sync')
