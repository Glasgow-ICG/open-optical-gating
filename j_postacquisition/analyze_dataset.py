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
from image_saving import *

def AnalyzeDataset(basePath, imageRangeList, annotationList, downsampling = 2, numSamplesPerPeriod = 60, earlyAnnotationTruncation = -1, source = 'Brightfield - Prosilica', periodRange = np.arange(23, 37, 0.1), plotAllPeriods=False, alpha=10, cropRect=None, cropFollowsZ=None, maxOffsetToConsider=None, applyDriftCorrection=True, annotationTimeShifts=None, interpolationDistanceBetweenSequences=0, rollingAverage=False, sourceTimestampKey='time_received', fluorTimestampKey='time_exposed', filenameFormat='%06d.tif'):
    # Load the images from disk (and determine the average period)
    #
    # basePath:              String giving directory containing a recorded dataset. Should NOT include the actual camera folder e.g. "Brightfield"
    # imageRangeList:        List of tuples specifying first and last frame index of a batch of frames to be processed
    #                        e.g. [(100, 200), (1042, 2010)]
    #                        although normally you would just specify a single range e.g. [(10000, 13984)]
    # annotationList:        List of channel folders for which you ultimately want the plist edited to add phase information
    # downsampling:          Brightfield frames will be downsampled by this factor before processing.
    #                        a factor of 2 seems to give pretty good results, and is a lot faster than no downsampling (downsampling=1)
    # numSamplesPerPeriod:   Number of units to break up the period into. Default value works well for 80fps datasets
    # earlyAnnotationTruncation: Normally leave this as -1. If set to a positive value we stop editing plists before the whole sequence has been processed
    # source:                Channel folder containing the (usually brightfield) channel to use to determine the heart phase
    # periodRange:           Numpy array specifying the candidate periods to use. THIS IS IMPORTANT TO GET RIGHT - see notes below.
    # plotAllPeriods:        Debug diagnostic flag that will plot lots of graphs related to our period-determination calculations
    # alpha:                 Factor used in the period analysis (usually leave as default value)
    # cropRect:              Optional rectangle specifying cropping of the data used for sync analysis
    #                        Format is (x1, x2, y1 y2), measured from the top left of the image
    # cropFollowsZ:          Crop rect is adjusted according to the z coordinate in the metadata (useful for side view z scans)
    #                        Values: None=ignore, 'x'=move cropRect in x, 'y'=move cropRect in y
    #                        For now, the code assumes we are scanning in a positive z direction, and it will increase the crop coordinates as z increases.
    # maxOffsetToConsider:   Factor controlling how we compare different beats for phase alignment.
    #                        Unless the actual focus of brightfield images is changing, this should normally be left as  None
    # applyDriftCorrection:  Should we make a somewhat-crude analysis of how much the brightfield images have drifted, and attempt to correct for that?
    #                        Drift will be present if doing a z scan with focus correction, or imaging from the side using the sheet launch objective.
    # annotationTimeShifts:  Adjustment that should be applied to the timestamps read from the metadata for the data channels in annotationList.
    #                        If present, this should be a list of numbers the same length as annotationList.
    #                        Positive means that the channel to be annotated (usually a fluorescence channel) has a later timestamp in its metadata
    #                        than the equivalent timestamp for a reference (brightfield) frame that in reality finished exposing at exactly the same moment.
    #                        That is the normal way round we would expect things to be (fluor cameras are generally slower to read out).
    #                        For normal acquisition, the time shift that should be applied here is the readout time for the QI camera
    #                        (1/max-framerate-for-short-exposures) minus the readout time for the PS camera.
    #                        Extra thought is required for PIV frame pairs - this scheme will give correct phase values for the laser pulse sent
    #                        in the *first* of a PIV frame pair (assuming this is close to the end of the camera exposure period),
    #                        but the calculated phase value for the second of a frame pair cannot be trusted.
    #                        NOTE: new datasets are now recording a plist key containing the actual exposure time for QI cameras
    #                        (as opposed to frame data received) which reduces the severity of the timeshift issue.
    # interpolationDistanceBetweenSequences: Original behaviour was to leave phases undefined between our 4pi image sections. However, image sections are normally
    #                        very close to each other, so it makes sense to interpolate across the gaps rather than leaving phases undefined.
    #                        This variable defines the maximum number of frames-worth of gaps in timestamp over which we will happily interpolate.
    #                        Default is 0 just because this was the original behaviour (no interpolation across gaps)
    #                        A small value (e.g. 4) will bridge across small gaps but will still avoid interpolating across anomalously large gaps due to actual breaks in the sequence
    # rollingAverage:        A special flag useful only for processing compressive photodiode data
    # sourceTimestampKey:    Key to look for in brightfield channel plists, to use as a timestamp. Note that for the PS camera the best we can do is time_received
    # fluorTimestampKey:     Key to look for in the fluorescence channel plists, to use as a timestamp. Note that old datasets do not have 'time_exposed'
    #                        TODO: at the moment I use this for all channels I want to annotate, which means we cannot do the brightfield channel if we use 'time_exposed'
    # filenameFormat:        Gives format for each filename in the image sequence (see usage in code below)
    #
    #    Before running the analysis, you should have a fairly good estimate of what the heart period (in frames) is.
    #    This can be done by examining the dataset in MovieBuilder or some other means.
    #    Then specify a numpy array of values to be tried (see scripts for examples of this).
    #    You should try and specify a fairly tight range, but that range must encompass ALL beat periods that
    #    may be present in the dataset.
    #    When you run an analysis script, it will print out the min and max period that were actually observed
    #    in the data. If these are too close to the edges of the range you originally specified, you should
    #    stop the script and rerun with a wider range.
    #    On the other hand, though, if you use a range that is too generous then there is a risk the code will
    #    mis-calculate some of the periods!
    #
    #    After running the analysis, it is important to check whether the synchronized output is convincing or not -
    #    there is a risk that the script has not (for whatever reason) managed to do a good job of phase recovery.
    #    In the case of a z stack, this can be done fairly easily by viewing the brightfield or fluor images
    #    in the StackViewer program on OS X. Alternatively (and on any OS), take a set of brightfield images
    #    with phases, loaded using LoadImages and friends, and then call:
    #    SaveImagesToFolder(SortImagesByPhase(images), '/Path/to/destination/folder')
    #    and check that the images look like they are in phase order. (In the case of a z scan, there will
    #    be some jumping around due to drift etc, but you should get a fairly clear sense that it is working).
    #
    
    # Load brightfield images for analysis
    pathFormat = '%s/'+source+'/'+filenameFormat
    if (imageRangeList is None):
        (images, averagePeriod) = LoadAllImages(basePath+'/'+source, downsampleFactor=downsampling, periodRange=periodRange, plotAllPeriods=plotAllPeriods, cropRect=cropRect, cropFollowsZ=cropFollowsZ, timestampKey=sourceTimestampKey)
    else:
        assert(len(imageRangeList) > 0)
        (firstImage, lastImage) = imageRangeList[0]
        (images, averagePeriod) = LoadImages(basePath, pathFormat, firstImage, lastImage - firstImage + 1, downsampleFactor=downsampling, periodRange=periodRange, plotAllPeriods=plotAllPeriods, cropRect=cropRect, cropFollowsZ=cropFollowsZ, timestampKey=sourceTimestampKey)
        for i in range(1, len(imageRangeList)):
            (firstImage, lastImage) = imageRangeList[i]
            lastIndex = images[-1].frameIndex
            (im2, dummy) = LoadImages(basePath, pathFormat, firstImage, lastImage - firstImage + 1, downsampleFactor=downsampling, periodRange=periodRange, plotAllPeriods=plotAllPeriods, frameIndexStart=lastIndex+1, cropRect=cropRect, cropFollowsZ=cropFollowsZ, timestampKey=sourceTimestampKey)
            images = np.append(images, im2)
            del im2

    # Temp: apply rolling 4-point average for the images (to deal with noise on the raw compressive images)
    if (rollingAverage):
        for i in range(len(images)-3):
            images[i].image = (images[i].image + images[i+1].image + images[i+2].image + images[i+3].image) / 4.0
            images[i].timestamp = (images[i].timestamp + images[i+1].timestamp + images[i+2].timestamp + images[i+3].timestamp) / 4.0



    # Looking at the scores when determining the period, there is a clear drop-off in score at higher values of period
    # I suspect this is because if time-sequential frames are adjacent in the phase-wrapped version
    # then the overall penalties are much lower (adjacent frames are more similar than frames one beat apart, due to RBCs etc)
    # This means that our candidate period bracket cannot be too generous.
    # For the "slow heart belly scan" dataset, a range of (20, 50) finds 50-ish for sequence 10, instead of 30ish.
    
    # Split into two-period chunks for phase alignment.
    # (doing two periods because Liebling's approach typically requires multiple periods in one chunk in order to be able to determine the period.
    # This could perhaps be improved using my method for determining the period in the first place.
    (imageSections, sectionPeriods) = SplitIntoSections(images, averagePeriod, periodRange=periodRange, plotAllPeriods=plotAllPeriods, alpha=alpha)

    # Resample each image section into exactly the same number of frames (as per Liebling's papers)
    print('period range', np.min(sectionPeriods), 'to', np.max(sectionPeriods))
    print (len(imageSections), 'image sections')
    resampledImageSections = ResampleUniformly(imageSections, sectionPeriods, numSamplesPerPeriod)
    
    # Attempt to free up memory by freeing the raw images - we don't need them any more
    del images
    del imageSections

    # Optional: crude attempt to correct for xy drift in the brightfield images
    # (which will occur due to focus correction as well as due to actual fish motion)
    if applyDriftCorrection:
        driftInset = 10     # TODO: Effectively this is the maximum drift we can handle. Needs to be coded properly!
        sequenceDrifts = CorrectForDrift(resampledImageSections, numSamplesPerPeriod, inset=driftInset)
    else:
        driftInset = 0
        sequenceDrifts = [(0, 0)] * len(resampledImageSections)

    # Determine the relative shifts between pairs of image sections
    shifts = GetShifts(resampledImageSections, sectionPeriods, sequenceDrifts, driftInset, numSamplesPerPeriod, maxOffsetToConsider)
    print(shifts)
    # Turn these relative shifts into one self-consistent global set of absolute shifts
    (globalShiftSolution, adjustedShifts, adjacentSolution, res, adjRes) = MakeShiftsSelfConsistent(shifts, len(resampledImageSections), numSamplesPerPeriod)
    
    #print (globalShiftSolution - adjacentSolution)
    
    if False:
        for i in range(len(shifts)):
            # Look at how each measured shift compares with the solution we obtained
            (i, j, sh, sc) = shifts[i]
            print(i, j, sh, sc, (globalShiftSolution[j]-globalShiftSolution[i])%numSamplesPerPeriod, (adjacentSolution[j]-adjacentSolution[i])%numSamplesPerPeriod)

    # Look at how the global solution differs from the adjacent solution
    # (bearing in mind that we don't necessarily know which one to trust!)
    # Note that it would be informative to work with a long video that truly is
    # just focused on a single plane. That would help test how much the shifts
    # will naturally wander due to random variation.
    # sqrt(n) random walk would suggest wandering 17 in 300, which is about what I see, in fact.
    # Another potentially interesting thing to do would be to do a z scan and then retreat
    # back to where we started (if this can be done smoothly), and see how much the phase has drifted.
    # Yet another would be to run the same analysis but with different resampling,
    # and see how similar the results are.
    #plt.plot(((adjacentSolution - globalShiftSolution + numSamplesPerPeriod/2.0) % numSamplesPerPeriod) - numSamplesPerPeriod/2.0)

    #print (adjacentSolution[50:80] % numSamplesPerPeriod)
    #print (globalShiftSolution[50:80] % numSamplesPerPeriod)
    #print (((adjacentSolution - globalShiftSolution)[50:80]) % numSamplesPerPeriod)


    # Define phases for resampled image sections
    # These give our definitive known time/phase mapping
    (knownTimes, knownPhases) = DefinePhaseForSequence(resampledImageSections, globalShiftSolution, numSamplesPerPeriod, plotIt = True, interpolationDistanceBetweenSequences=interpolationDistanceBetweenSequences)
    # Fill in the phases on the original brightfield images (because why not)
    if False:
        # "Why not"? We may run out of memory on 32-bit python!
        (dummy1, dummy2) = DefinePhaseForSequence(imageSections, globalShiftSolution, numSamplesPerPeriod)

    # Further attempts to free up memory
    del resampledImageSections

    for i in range(len(annotationList)):
        # Annotate the fluorescence channels by reading the timestamps and assigning a phase by interpolating within knownTimes/knownPhases.
        # This is easy if the timebases for all channels match. However with SPIM data we are forced to use the times at which the frame data
        # is received on the mac, since this is the only timebase we have that is universal across all cameras.
        # That time represents the time after the full image exposure has completed AND the frame data has been fully transferred to the computer.
        # Thus it would often be appropriate to adjust the time values to compensate for this.
        # Here we (temporarily) add the adjustment to the *reference* [brightfield] times, which is consistent with the convention defined in this function header,
        # that a positive time shift represents a fluor frame ending up with a larger timestamp value than a brightfield frame acquired at the same actual moment in time.
        annotationFolder = annotationList[i]
        if annotationTimeShifts is not None:
            annotationTimeShift = annotationTimeShifts[i]
        else:
            annotationTimeShift = 0
        AnnotateFluorChannel('%s/%s' % (basePath, annotationFolder), knownTimes + annotationTimeShift, knownPhases, earlyAnnotationTruncation, fluorTimestampKey=fluorTimestampKey)

    return (knownTimes, knownPhases, globalShiftSolution, res, shifts, sequenceDrifts)
