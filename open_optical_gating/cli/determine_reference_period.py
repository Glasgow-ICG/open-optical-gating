"""Functions to establish a reference heartbeat/period
as used from prospective optical gating."""

# Module imports
import os
import numpy as np
from loguru import logger
from datetime import datetime
import j_py_sad_correlation as jps
# See comment in pyproject.toml for why we have to try both of these
try:
    import skimage.io as tiffio
    using_skimage = True
except:
    import tifffile as tiffio
    using_skimage = False

# Local
from . import prospective_optical_gating as pog

# Define an important constant related to padding in the reference frames.
# Really the only reason this parameter exists at all in the original C code is to self-document all the +-2 arithmetic that would otherwise appear.
# In the C code it is declared as a const int.
# Note that it should definitely not be changed on-the-fly, or algorithm behaviour will be "undefined"!
numExtraRefFrames = 2

class ReferenceSequenceManager:
    def __init__(self, ref_settings):
        self.frame_history = []
        self.period_history = []
        self.ref_frames = None
        self.ref_settings = ref_settings
        if (("bootstrap_drift_search_range" in ref_settings) and
            (ref_settings["bootstrap_drift_search_range"] > 0)
           ):
            self.drift_bootstrap_needed = True
        else:
            self.drift_bootstrap_needed = False
        self.drift = (0, 0)
    
    def establish_period_from_frames(self, pixel_array):
        """ Attempt to establish a period from the frame history,
            including the new frame represented by 'pixel_array'.
            
            Returns: True/False depending on if we have successfully identified a one-heartbeat reference sequence
        """
        # Add the new frame to our history buffer
        self.frame_history.append(pixel_array)
    
        # Impose an upper limit on the buffer length, to protect against performance degradation
        # in cases where we are not succeeding in identifying a period.
        # That limit is defined in terms of how many seconds of frame data we have,
        # relative to the minimum heart rate (in Hz) that we are configured to expect.
        ref_buffer_duration = (self.frame_history[-1].metadata["timestamp"] - self.frame_history[0].metadata["timestamp"])
        while (ref_buffer_duration > 1.0/self.ref_settings["min_heart_rate_hz"]):
            # I have coded this as a while loop, but we would normally expect to only trim one frame at a time
            logger.debug("Trimming buffer from duration {0} to {1}".format(ref_buffer_duration, 1.0/self.ref_settings["min_heart_rate_hz"]))
            del self.frame_history[0]
            ref_buffer_duration = (self.frame_history[-1].metadata["timestamp"] - self.frame_history[0].metadata["timestamp"])
        
        return establish(self.frame_history, self.period_history, self.ref_settings)

    def set_ref_frames(self, ref_frames, period_to_use):
        # Note that ref_frames is probably a list, and this is helpful because it means we could still access
        # the PixelArray metadata up to this point.
        # However, long-term we want to store a 3D array because that is what OGA expects to work with.
        # We therefore make that conversion here
        self.ref_frames = np.array(ref_frames)
        self.ref_period = period_to_use

    def pick_good_target_and_barrier_frames(self):
        """ Function to automatically identify a stable target phase and barrier frame.
            
            Looks through 'ref_frames' to identify a consistent point in the
            heart cycle (i.e. attempt to identify approximately the same absolute
            phase every time we establish sync, rather than just picking a random
            point in the cycle as our initial target sync phase).
         
            NOTE: this is not used when using the adaptive algorithms. Rather,
            this is implemented for our convenience when first setting up the sync.
            Most of the time, we find that the phase selected by this function is a
            reasonable one to synchronize to (whereas some phases such as the
            refractory phases between beats) may be less reliable if they are
            chosen as the target sync phase.
         
            The function uses various heuristics in the hope of identifying a
            consistent point in the heart cycle. For side-on orientations (as seen by
            the brightfield camera) of the fish, this does a reasonable job of
            identifying a point in the cycle that gives a fairly good sync.
            For belly-on orientations (as seen by the brightfield camera), it
            does not always work so well, and manual intervention is more likely
            to be needed to pick an appropriate target sync phase.

            Returns:
                target_frame_without_padding    float   Index into the reference sequence array that represents
                                                         the chosen target frame, but indexed such that 0 is the
                                                         frame that represents 0 phase. This is not the actual
                                                         frame 0 in the array, due to the additional padding we include.
                barrier_frame_without_padding   float   Equivalent index for the barrier frame
        """

        # First compare each frame in our list with the previous one
        # Note that this code assumes numExtraRefFrames>0 (which it certainly should be!)
        deltas_without_padding = np.zeros(
            (len(self.ref_frames) - 2 * numExtraRefFrames), dtype=np.int64,
        )
        for i in np.arange(len(self.ref_frames) - 2 * numExtraRefFrames,):
            deltas_without_padding[i] = jps.sad_correlation(
                self.ref_frames[i + numExtraRefFrames],
                self.ref_frames[i + numExtraRefFrames + 1],
            )

        min_pos_without_padding = np.argmin(deltas_without_padding)
        max_pos_without_padding = np.argmax(deltas_without_padding)
        min_delta = deltas_without_padding.min()
        max_delta = deltas_without_padding.max()
        target_frame_without_padding = 0

        # The greatest diff (most rapid change) tends to come soon after the region of minimum change.
        # The greatest diff tends to be a sharp peak (whereas, almost by definition, the minimum change is broader)/
        # We use the greatest diff as our "universal"(ish) reference point, but default to a phase 1/3 of a period after it,
        # so that we get a good clear run-up to it with well-defined phases in our history.

        # Use v-fitting to find a sub-frame estimate of the maximum
        if (max_pos_without_padding <= 0) or (max_pos_without_padding == deltas_without_padding.size - 1):
            # It looks as if the best position is right at the start or end of the dataset.
            # Presumably due to a slightly glitch it's possible that the true minimum lies just outside the dataset.
            # However, in pathological datasets there could be no minimum in easy reach at all,
            # so we've got to give up at some point.
            # Therefore, if we hit this condition, we just decide the best offset is 0.0.
            # In sensible cases, that will be very close to optimum.
            # In messy cases, this entire function is going to do unpredictable things anyway, so who cares!
            target_frame_without_padding = 0
        else:
            target_frame_without_padding = (
                max_pos_without_padding
                + v_fitting(
                    -deltas_without_padding[max_pos_without_padding - 1],
                    -deltas_without_padding[max_pos_without_padding],
                    -deltas_without_padding[max_pos_without_padding + 1],
                )[0]
            )

        # Now shift that forwards by 1/3 of a period but wrap if this is in the next beat
        target_frame_without_padding = (
            target_frame_without_padding + (self.ref_period / 3.0)
        ) % self.ref_period

        # We also identify a point soon after the region of minimum change ends.
        # We look for the diffs to rise past their midpoint between min and max (which is expected to happen soon).
        # We should try not to fit our history back beyond that, in order to avoid trying to fit to
        # the (unpredictable) refractory period of the heart
        barrier_frame_without_padding = min_pos_without_padding
        while (
            deltas_without_padding[barrier_frame_without_padding]
            < (min_delta + max_delta) / 2
        ):
            barrier_frame_without_padding = (barrier_frame_without_padding + 1) % int(self.ref_period)
            if barrier_frame_without_padding == min_pos_without_padding:
                # This means I have done a complete loop and found no delta less than the average of the min and max deltas
                # Probably means my frames are all empty so my diffs are all zeros
                logger.debug(
                    "Search for a barrier frame has failed: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}",
                    min_pos_without_padding,
                    min_delta,
                    max_pos_without_padding,
                    max_delta,
                    target_frame_without_padding,
                    barrier_frame_without_padding,
                    self.ref_period,
                    numExtraRefFrames,
                )
                break

        # Return new value for target_frame and barrier_frame
        logger.debug(
            "Identified recommended target frame of {0} and barrier frame of {1} (both without padding).",
            target_frame_without_padding,
            barrier_frame_without_padding,
        )
        return target_frame_without_padding, barrier_frame_without_padding
    
    def set_target_and_barrier_frame(self, target, barrier):
        self.targetFrameNum = target
        self.barrierFrameNum = barrier
        self.targetSyncPhase = 2*np.pi * (target / self.ref_period)

    def save_ref_sequence(self, ref_seq_dir):
        # JT TODO: should also save period.txt as I do in the C code.
        save_period(self.ref_frames, self.ref_settings["reference_sequence_dir"])

    def identify_phase_for_frame(self, frame):
        """ Phase match a new frame based on a reference period.
            
            Parameters:
                frame               array-like      2D frame pixel data for our most recently-received frame
            Returns:
                matched_phase       float           Phase (0-2π) associated with the best matching location in the reference frames array
                SADs                ndarray         1D sum of absolute differences between frame and each reference_frames[t,...]
        """
        if self.drift_bootstrap_needed:
            # We are required to bootstrap the drift estimate with a slow search over a wider area
            logger.trace("Bootstrapping drift correction")
            r = self.ref_settings["bootstrap_drift_search_range"]
            self.drift = get_drift_estimate(frame, self.ref_frames, dxRange=range(-r,r+1,3), dyRange=range(-r,r+1,3))
            logger.trace("Bootstrapping drift correction: {0)", self.drift)
            self.drift_bootstrap_needed = False

        # Apply drift correction, identifying a crop rect for the frame and/or reference frames,
        # representing the area intersection between them once drift is accounted for.
        logger.trace("Applying drift correction of {0}", self.drift)
        
        dx, dy = self.drift
        rectF = [0, frame.shape[0], 0, frame.shape[1]]  # X1,X2,Y1,Y2
        rect = [
            0,
            self.ref_frames[0].shape[0],
            0,
            self.ref_frames[0].shape[1],
        ]  # X1,X2,Y1,Y2

        if dx <= 0:
            rectF[0] = -dx
            rect[1] = rect[1] + dx
        else:
            rectF[1] = rectF[1] - dx
            rect[0] = dx
        if dy <= 0:
            rectF[2] = -dy
            rect[3] = rect[3] + dy
        else:
            rectF[3] = rectF[3] - dy
            rect[2] = +dy

        frame_cropped = frame[rectF[0] : rectF[1], rectF[2] : rectF[3]]
        reference_frames_cropped = [
            f[rect[0] : rect[1], rect[2] : rect[3]] for f in self.ref_frames
        ]

        # Calculate SADs
        logger.trace(
            "Reference frame dtypes: {0} and {1}", frame.dtype, self.ref_frames[0].dtype
        )
        logger.trace(
            "Reference frame shapes: {0}->{1} and {2}->{3}", frame.shape, frame_cropped.shape,
            self.ref_frames[0].shape, reference_frames_cropped[0].shape
        )
        SADs = jps.sad_with_references(frame_cropped, reference_frames_cropped)
        logger.trace("SADs: {0}", SADs)
        
        # Identify best match between 'frame' and the reference frame sequence
        matched_phase = find_subframe_minimum(SADs, self.ref_period)
        logger.debug("Found best matching phase to be {0}", matched_phase)

        # Update current drift estimate in the settings dictionary
        self.drift = update_drift_estimate(frame, self.ref_frames[np.argmin(SADs)], (dx, dy))
        logger.debug("Drift correction updated to {0}", self.drift)

        return (matched_phase, SADs)

def establish(sequence, period_history, ref_settings, require_stable_history=True):
    """ Attempt to establish a reference period from a sequence of recently-received frames.
        Parameters:
            sequence        list of PixelArray objects  Sequence of recently-received frame pixel arrays (in chronological order)
            period_history  list of float               Values of period calculated for previous frames (which we will append to)
            ref_settings    dict                        Parameters controlling the sync algorithms
            require_stable_history  bool                Do we require a stable history of similar periods before we consider accepting this one?
        Returns:
            List of frame pixel arrays that form the reference sequence (or None).
            Exact (noninteger) period for the reference sequence
    """
    start, stop, periodToUse = establish_indices(sequence, period_history, ref_settings, require_stable_history)
    if (start is not None) and (stop is not None):
        referenceFrames = sequence[start:stop]
    else:
        referenceFrames = None

    return referenceFrames, periodToUse

def establish_indices(sequence, period_history, ref_settings, require_stable_history=True):
    """ Establish the list indices representing a reference period, from a given input sequence.
        Parameters: see header comment for establish(), above
        Returns:
            List of indices that form the reference sequence (or None).
    """
    logger.debug("Attempting to determine reference period.")
    if len(sequence) > 1:
        frame = sequence[-1]
        pastFrames = sequence[:-1]

        # Calculate Diffs between this frame and previous frames in the sequence
        diffs = jps.sad_with_references(frame, pastFrames)
        logger.trace("SADs: {0}", diffs)

        # Calculate Period based on these Diffs
        period = calculate_period_length(diffs, ref_settings["minPeriod"], ref_settings["lowerThresholdFactor"], ref_settings["upperThresholdFactor"])
        if period != -1:
            period_history.append(period)

        # If we have a valid period, extract the frame indices associated with this period, and return them
        # The conditions here are empirical ones to protect against glitches where the heuristic
        # period-determination algorithm finds an anomalously short period.
        # JT TODO: The three conditions on the period history seem to be pretty similar/redundant. I wrote these many years ago,
        #  and have just left them as they "ain't broke". They should really be tidied up though.
        #  One thing I can say is that the reason for the *two* tests for >6 have to do with the fact that
        #  we are establishing the period based on looking back from the *most recent* frame, but then actually
        #  and up taking a period from a few frames earlier, since we also need to incorporate some extra padding frames.
        #  That logic could definitely be improved and tidied up - we should probably just
        #  look for a period starting numExtraRefFrames from the end of the sequence...
        # TODO: JT writes: logically these tests should probably be in calculate_period_length, rather than here
        history_stable = (len(period_history) >= (5 + (2 * numExtraRefFrames))
                            and (len(period_history) - 1 - numExtraRefFrames) > 0
                            and (period_history[-1 - numExtraRefFrames]) > 6)
        if (
            period != -1
            and period > 6
            and ((require_stable_history == False) or (history_stable))
        ):
            # We pick out a recent period from period_history.
            # Note that we don't use the very most recent value, because when we pick our reference frames
            # we will pad them with numExtraRefFrames at either end. We pick the period value that
            # pertains to the range of frames that we will actually use
            # for the central "unpadded" range of our reference frames.
            periodToUse = period_history[-1 - numExtraRefFrames]
            logger.debug("Found a period I'm happy with: {0}".format(periodToUse))

            # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
            numRefs = int(periodToUse + 1) + (2 * numExtraRefFrames)

            # return start, stop, period
            logger.debug(
                "Start index: {0}; Stop index: {1}; Period {2}",
                len(pastFrames) - numRefs,
                len(pastFrames),
                periodToUse
            )
            return len(pastFrames) - numRefs, len(pastFrames), periodToUse

    logger.debug("I didn't find a period I'm happy with!")
    return None, None, None

def calculate_period_length(diffs, minPeriod=5, lowerThresholdFactor=0.5, upperThresholdFactor=0.75):
    """ Attempt to determine the period of one heartbeat, from the diffs array provided. The period will be measured backwards from the most recent frame in the array
        Parameters:
            diffs    ndarray    Diffs between latest frame and previously-received frames
        Returns:
            Period, or -1 if no period found
    """

    # Calculate the heart period (with sub-frame interpolation) based on a provided list of comparisons between the current frame and previous frames.
    bestMatchPeriod = None

    # Unlike JTs codes, the following currently only supports determining the period for a *one* beat sequence.
    # It therefore also only supports determining a period which ends with the final frame in the diffs sequence.
    if diffs.size < 2:
        logger.trace("Not enough diffs, returning -1")
        return -1

    # initialise search parameters for last diff
    score = diffs[diffs.size - 1]
    minScore = score
    maxScore = score
    totalScore = score
    meanScore = score
    minSinceMax = score
    deltaForMinSinceMax = 0
    stage = 1
    numScores = 1
    got = False

    for d in range(minPeriod, diffs.size+1):
        score = diffs[diffs.size - d]
        # got, values = gotScoreForDelta(score, d, values)

        totalScore += score
        numScores += 1

        lowerThresholdScore = minScore + (maxScore - minScore) * lowerThresholdFactor
        upperThresholdScore = minScore + (maxScore - minScore) * upperThresholdFactor
        logger.trace(
            "d:{0};\tLower Threshold:\t{1:.4f};\tUpper Threshold:\t{2:.4f}",
            d,
            lowerThresholdScore,
            upperThresholdScore,
        )

        if score < lowerThresholdScore and stage == 1:
            logger.trace("Stage 1: Under lower threshold; Moving to stage 2")
            stage = 2

        if score > upperThresholdScore and stage == 2:
            # TODO: speak to JT about the 'final condition'
            logger.trace(
                "Stage 2: Above upper threshold; Returning period of {0}",
                deltaForMinSinceMax,
            )
            stage = 3
            got = True
            break

        if score > maxScore:
            logger.trace(
                "New max score: {0} > {1}. Resetting to stage 1.", score, maxScore
            )
            maxScore = score
            minSinceMax = score
            deltaForMinSinceMax = d
            stage = 1
        elif score != 0 and (minScore == 0 or score < minScore):
            logger.trace("New minimum score of {0}", score)
            minScore = score

        if score < minSinceMax:
            logger.trace(
                "New minimum score ({0}) since maximum of {1}", score, maxScore
            )
            minSinceMax = score
            deltaForMinSinceMax = d

        # Note this is only updated AFTER we have done the other processing (i.e. the mean score used does NOT include the current delta)
        meanScore = totalScore / numScores

    if got:
        bestMatchPeriod = deltaForMinSinceMax

    if bestMatchPeriod is None:
        logger.debug("I didn't find a whole period, returning -1")
        return -1

    bestMatchEntry = diffs.size - bestMatchPeriod

    interpolatedMatchEntry = (
        bestMatchEntry
        + v_fitting(
            diffs[bestMatchEntry - 1], diffs[bestMatchEntry], diffs[bestMatchEntry + 1]
        )[0]
    )

    return diffs.size - interpolatedMatchEntry

def save_period(reference_sequence, parent_dir="~/", prefix="REF-"):
    """Function to save a reference period in an ISO format time-stamped folder with a parent_dir.
        Parameters:
            reference_sequence    ndarray     t by x by y 3d array of reference frames
            parent_dir          string      parent directory within which to store the period
    """
    dt = datetime.now().strftime(prefix+"%Y-%m-%dT%H%M%S")
    os.makedirs(os.path.join(parent_dir, dt), exist_ok=True)

    # Saves the period
    for i, frame in enumerate(reference_sequence):
        if using_skimage:
            # skimage will output a warning if it thinks we are passing it a low-contrast image
            tiffio.imsave(os.path.join(parent_dir, dt, "{0:03d}.tiff".format(i)), frame, check_contrast = False)
        else:
            # ... but tifffile does not accept that parameter, so we need separate code here
            tiffio.imsave(os.path.join(parent_dir, dt, "{0:03d}.tiff".format(i)), frame)

    logger.debug("Saved frames to \"{0}\"", dt)

def update_drift_estimate(frame0, bestMatch0, drift0):
    """ Determine an updated estimate of the sample drift.
        We try changing the drift value by ±1 in x and y.
        This just calls through to the more general function get_drift_estimate()
        
        Parameters:
            frame0         array-like      2D frame pixel data for our most recently-received frame
            bestMatch0     array-like      2D frame pixel data for the best match within our reference sequence
            drift0         (int,int)       Previously-estimated drift parameters
        Returns
            new_drift      (int,int)       New drift parameters
        """
    return get_drift_estimate(frame0, [bestMatch0], dxRange=range(drift0[0]-1, drift0[0]+2), dyRange=range(drift0[1]-1, drift0[1]+2))

def get_drift_estimate(frame, refs, matching_frame=None, dxRange=range(-30,31,3), dyRange=range(-30,31,3)):
    """ Determine an initial estimate of the sample drift.
        We do this by trying a range of variations on the relative shift between frame0 and the best-matching frame in the reference sequence.
        
        Parameters:
            frame          array-like      2D frame pixel data for the frame we should use
            refs           list of arrays  List of 2D reference frame pixel data that we should search within
            matching_frame int             Entry within reference frames that is the best match to 'frame',
                                        or None if we don't know what the best match is yet
            dxRange        list of int     Candidate x shifts to consider
            dyRange        list of int     Candidate y shifts to consider
        
        Returns:
            new_drift      (int,int)       New drift parameters
        """
    # frame0 and the images in 'refs' must be numpy arrays of the same size
    assert frame.shape == refs[0].shape
    
    # Identify region within bestMatch that we will use for comparison.
    # The logic here basically follows that in phase_matching, but allows for extra slop space
    # since we will be evaluating various different candidate drifts
    inset = np.maximum(np.max(np.abs(dxRange)), np.max(np.abs(dyRange)))
    rect = [
            inset,
            frame.shape[0] - inset,
            inset,
            frame.shape[1] - inset,
            ]  # X1,X2,Y1,Y2
            
    candidateShifts = []
    for _dx in dxRange:
        for _dy in dyRange:
            candidateShifts += [(_dx,_dy)]

    if matching_frame is None:
        ref_index_to_consider = range(0, len(refs))
    else:
        ref_index_to_consider = [matching_frame]

    # Build up a list of frames, each representing a window into frame with slightly different drift offsets
    frames = []
    for shft in candidateShifts:
        dxp = shft[0]
        dyp = shft[1]
        
        # Adjust for drift and shift
        rectF = np.copy(rect)
        rectF[0] -= dxp
        rectF[1] -= dxp
        rectF[2] -= dyp
        rectF[3] -= dyp
        frames.append(frame[rectF[0] : rectF[1], rectF[2] : rectF[3]])

    # Compare all these candidate shifted images against each of the candidate reference frame(s) in turn
    # Our aim is to find the best-matching shift from within the search space
    best = 1e200
    for r in ref_index_to_consider:
        sad = jps.sad_with_references(refs[r][rect[0] : rect[1], rect[2] : rect[3]], frames)
        smallest = np.min(sad)
        if (smallest < best):
            bestShiftPos = np.argmin(sad)
            best = smallest

    return (candidateShifts[bestShiftPos][0],
            candidateShifts[bestShiftPos][1])

def find_subframe_minimum(diffs, reference_period):
    """ Identify the phase by identifying the minimum value (to sub-frame accuracy) within a list of diffs (which includes padding frames).
        The aim is to find the position of the minimum within the "main" frames,
        but we may have to extend to the padding frames if the minimum is right at one end of the "main" frames.
        
        Parameters:
            diffs             list/array-like   Sequence of differences between current frame and a set of reference frames
            reference_period  float             Period for the reference frames that "diffs" was comparing against
        Returns:
            float value of phase (0-2π)
    """
    # Search for lowest value within the "main" frames.
    # Note that this code assumes numExtraRefFrames>0 (which it certainly should be!)
    bestScorePos = np.argmin(
        diffs[numExtraRefFrames : -numExtraRefFrames]
    )
    bestScorePos = bestScorePos + numExtraRefFrames

    # Sub-pixel fitting
    if ((diffs[bestScorePos - 1] < diffs[bestScorePos]) or
        (diffs[bestScorePos + 1] < diffs[bestScorePos])):
        # No V minimum was found within the "main" reference frames (ignoring padding)
        # If we get here then there is no actual "v" minimum to be found within the range of our
        # main reference frames. It is probably to be found just outside, and we could
        # potentially attempt to fit it anyway. However this should be an unusual occurrence,
        # and an assumption of a phase of 0 shouldn't be too far from the truth,
        # so to keep things simple and robust I'm going to leave it at this for now.
        # We just return a reference phase that corresponds to a phase of 0
        logger.debug("No minimum found - defaulting to phase=0")
        return 0.0
    else:
        # A minimum exists - do sub-frame interpolation on it
        interpolatedCorrection, _ = v_fitting(
            diffs[bestScorePos - 1], diffs[bestScorePos], diffs[bestScorePos + 1]
        )
        thisFrameReferencePos = bestScorePos + interpolatedCorrection
        # Convert phase to 2pi base
        return (
                2 * np.pi
                * (thisFrameReferencePos - numExtraRefFrames)
                / reference_period
               )  # rad

def v_fitting(y_1, y_2, y_3):
    # Fit using a symmetric 'V' function, to find the interpolated minimum for three datapoints y_1, y_2, y_3,
    # which are considered to be at coordinates x=-1, x=0 and x=+1
    if y_1 > y_3:
        x = 0.5 * (y_1 - y_3) / (y_1 - y_2)
        y = y_2 - x * (y_1 - y_2)
    else:
        x = 0.5 * (y_1 - y_3) / (y_3 - y_2)
        y = y_2 + x * (y_3 - y_2)

    return x, y

def u_fitting(y_1, y_2, y_3):
    # Fit using a symmetric 'U' function, to find the interpolated minimum for three datapoints y_1, y_2, y_3,
    # which are considered to be at coordinates x=-1, x=0 and x=+1
    c = y_2
    a = (y_1 + y_3 - 2*y_2) / 2
    b = y_3 - a - c
    x = -b / (2*a)
    y = a*x**2 + b*x + c
    return x, y, a, b, c

def u_fittingN(y):
    # Quadratic best fit to N datapoints, which [** inconsistently with respect to u/v_fitting() **]
    # are considered to be at coordinates x=0, 1, ...
    x = np.arange(len(y))
    def quadratic(x, a, b, c):
        return a*x**2 + b*x + c
    (a, b, c), cov = scipy.optimize.curve_fit(quadratic, x, y, p0=[0, 0, np.average(y)])
    x = -b / (2*a)
    y = quadratic(x, a, b, c)
    #    return x, y, *popt
    return x, y, a, b, c
