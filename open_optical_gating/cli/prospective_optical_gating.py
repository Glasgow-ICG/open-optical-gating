"""Module for real-time phase matching of brightfield sequences.
These codes are equivalent to the Objective-C codes in spim-interface."""

# Python imports
import numpy as np
from loguru import logger
import scipy.optimize

# Module imports
import j_py_sad_correlation as jps

# Local imports
from . import pog_settings as ps

# Define an important constant related to padding in the reference frames.
# Really the only reason this parameter exists at all in the original C code is to self-document all the +-2 arithmetic that would otherwise appear.
# In the C code it is declared as a const int.
# Note that it should definitely not be changed on-the-fly, or algorithm behaviour will be "undefined"
numExtraRefFrames = 2

def update_drift(frame0, bestMatch0, drift0):
    """ Determine an updated estimate of the sample drift.
        We do this by trying variations on the relative shift between frame0 and the best-matching frame in the reference sequence.
        
        Parameters:
            frame0         array-like      2D frame pixel data for our most recently-received frame
            bestMatch0     array-like      2D frame pixel data for the best match within our reference sequence
        Returns
            new_drift      (int,int)       New drift parameters
        """
    # frame0 and bestMatch0 must be numpy arrays of the same size
    assert frame0.shape == bestMatch0.shape

    # Start with the existing drift parameters in the settings dictionary
    dx, dy = drift0

    # Identify region within bestMatch that we will use for comparison.
    # The logic here basically follows that in phase_matching, but allows for extra slop space
    # since we will be evaluating various different candidate drifts
    rect = [
        abs(dx) + 1,
        frame0.shape[0] - abs(dx) - 1,
        abs(dy) + 1,
        frame0.shape[1] - abs(dy) - 1,
    ]  # X1,X2,Y1,Y2
    bestMatch = bestMatch0[rect[0] : rect[1], rect[2] : rect[3]]

    candidateShifts = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]

    # Build up a list of frames, each representing a window into frame0 with slightly different drift offsets
    frames = []
    for shft in candidateShifts:
        dxp = dx + shft[0]
        dyp = dy + shft[1]

        # Adjust for drift and shift
        rectF = np.copy(rect)
        rectF[0] -= dxp
        rectF[1] -= dxp
        rectF[2] -= dyp
        rectF[3] -= dyp

        frames.append(frame0[rectF[0] : rectF[1], rectF[2] : rectF[3]])

    # Compare all these candidate shifted images against the matching reference frame, and find the best-matching shift
    sad = jps.sad_with_references(bestMatch, frames)
    best = np.argmin(sad)

    return (dx + candidateShifts[best][0],
            dy + candidateShifts[best][1])


def subframe_phase_fitting(diffs, reference_period):
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
        logger.warning("No minimum found - defaulting to phase=0")
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

def identify_phase_with_drift(frame, reference_frames, reference_period, drift):
    """ Phase match a new frame based on a reference period.
        
        Parameters:
            frame               array-like      2D frame pixel data for our most recently-received frame
            reference_frames    array-like      3D (t by x by y) frame pixel data for our reference sequence
            reference_period    float           Period associated with reference_frames
            drift               (int,int)       Current drift value
        Returns:
            matched_phase       float           Phase (0-2π) associated with the best matching location in the reference frames array
            SADs                ndarray         1D sum of absolute differences between frame and each reference_frames[t,...]
            new_drift           (int,int)       Updated drift value
    """

    dx, dy = drift

    # Apply drift correction, identifying a crop rect for the frame and/or reference frames,
    # representing the area intersection between them once drift is accounted for.
    logger.info("Applying drift correction of ({0},{1})", dx, dy)
    rectF = [0, frame.shape[0], 0, frame.shape[1]]  # X1,X2,Y1,Y2
    rect = [
        0,
        reference_frames[0].shape[0],
        0,
        reference_frames[0].shape[1],
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
        f[rect[0] : rect[1], rect[2] : rect[3]] for f in reference_frames
    ]

    # Calculate SADs
    logger.trace(
        "Reference frame dtypes: {0} and {1}", frame.dtype, reference_frames[0].dtype
    )
    logger.trace(
        "Reference frame shapes: {0}->{1} and {2}->{3}", frame.shape, frame_cropped.shape,
        reference_frames[0].shape, reference_frames_cropped[0].shape
    )
    SADs = jps.sad_with_references(frame_cropped, reference_frames_cropped)
    logger.trace("SADs: {0}", SADs)

    # Identify best match between 'frame' and the reference frame sequence
    matched_phase = subframe_phase_fitting(SADs, reference_period)
    logger.debug("Found best matching phase to be {0}", matched_phase)

    # Update current drift estimate in the settings dictionary
    dx, dy = update_drift(frame, reference_frames[np.argmin(SADs)], (dx, dy))
    logger.info("Drift correction updated to ({0},{1})", dx, dy)

    return (matched_phase, SADs, (dx, dy))


def predict_trigger_wait(frame_history, pog_settings, fitBackToBarrier=True, framesForFit=None):
    """ Predict how long we need to wait until the heart is at the target phase we are triggering to.
        
        Parameters:
            frame_history           array-like  Nx3 array of [timestamp in seconds, phase, argmin(SAD)]
                                                 Phase (i.e. frame_history[:,1]) should be cumulative (i.e. phase-UNwrapped) phase in radians
            pog_settings            dict        Parameters controlling the sync algorithms
                                                 targetSyncPhase is expected to be in [0,2pi]
            fitBackToBarrier        bool        Should we use the "barrier frame" logic? (see determine_barrier_frames)
            framesForFit            int         Optional override value to fit to a specific number of frames from the past history.
                                                Note that the main purpose of this parameter is for use when we recursively call ourselves
        Returns:
            Time delay (in seconds) between when the current frame was
             acquired and when the trigger would need to be sent.
            Current heart rate (radians/sec), for information
        """

    # Deal with the barrier frame logic (if fitBackToBarrier is True):
    # Rather than fitting to a number of frames that depends on how far forward we are predicting,
    # fit to a number that depends on where in the cycle we are.
    # We try not to fit to the refractory period unless there really is no other data.
    # The intention of this is to fit to as much data as possible but only in the parts of the cycle
    # where the phase progression is highly predictable and linear with time.
    if framesForFit is None:
        if fitBackToBarrier:
            allowedToExtendNumberOfFittedPoints = False
            framesForFit = min(
                pog_settings["framesForFitLookup"][int(frame_history[-1, 2])],
                frame_history.shape[0],
            )
            logger.debug("Consider {0} past frames for prediction;", framesForFit)
        else:
            framesForFit = pog_settings["minFramesForFit"]
            allowedToExtendNumberOfFittedPoints = True

    if frame_history.shape[0] < framesForFit:
        logger.debug("Fit failed due to too few frames")
        return -1

    pastPhases = frame_history[-int(framesForFit):, :]

    # Perform a linear fit to the past phases. We will use this for our forward-prediction
    radsPerSec, alpha = np.polyfit(pastPhases[:, 0], pastPhases[:, 1], 1)

    logger.trace(pastPhases[:, 0])
    logger.trace(pastPhases[:, 1])

    logger.info("Linear fit with intersect {0} and gradient {1}", alpha, radsPerSec)
    if radsPerSec < 0:
        logger.warning(
            "Linear fit to unwrapped phases is negative! This is a problem for the trigger prediction."
        )
    elif radsPerSec == 0:
        logger.warning(
            "Linear fit to unwrapped phases is zero! This will be a problem for prediction (divByZero)."
        )

    # Use our linear fit to get a 'fitted' unwrapped phase for the latest frame
    # This should not rescue cases where, for some reason, the image-based
    # phase matching is erroneous.
    thisFramePhase = alpha + frame_history[-1, 0] * radsPerSec
    # Count how many total periods we have seen
    multiPhaseCounter = thisFramePhase // (2 * np.pi)
    # Determine how much of a cardiac cycle we have to wait till our target phase
    phaseToWait = (
        pog_settings["targetSyncPhase"] + (multiPhaseCounter * 2 * np.pi) - thisFramePhase
    )
    # c.f. function triggerAnticipationProcessing in SyncAnalyzer.mm
    # essentially this fixes for small backtracks in phase due to SAD imperfections.
    # If our computations so far suggest that our target phase is in the past, then
    # we add multiples of 2pi until we are targeting the same phase point in a future heartbeat.
    while phaseToWait < 0:
        phaseToWait += 2 * np.pi

    time_to_wait_seconds = phaseToWait / radsPerSec
    time_to_wait_seconds = max(time_to_wait_seconds, 0.0)

    logger.info(
        "Current time: {0};\tTime to wait: {1};",
        frame_history[-1, 0],
        time_to_wait_seconds,
    )
    logger.debug(
        "Current phase: {0};\tPhase to wait: {1};", thisFramePhase, phaseToWait,
    )
    logger.debug(
        "Target phase:{0};\tPredicted phase:{1};",
        pog_settings["targetSyncPhase"] + (multiPhaseCounter * 2 * np.pi),
        thisFramePhase + phaseToWait,
    )

    # Fixes sync error due to targetSyncPhase being 2pi greater than target phase (1e-3 is for floating point errors)
    if (
        thisFramePhase
        + phaseToWait
        - pog_settings["targetSyncPhase"]
        - (multiPhaseCounter * 2 * np.pi)
        > 2 * np.pi + 1e-3
    ):
        logger.warning(
            "Phase discrepency, trigger aborted. At {0} with wait {1} for target {2} [{3}]",
            thisFramePhase % (2 * np.pi),
            phaseToWait,
            pog_settings["targetSyncPhase"],
            thisFramePhase
            + phaseToWait
            - pog_settings["targetSyncPhase"]
            - (multiPhaseCounter * 2 * np.pi),
        )
        time_to_wait_seconds = 0.0

    # This logic catches cases where we are predicting a long way into the future using only a small number of datapoints.
    # That is likely to be error-prone, so (unless using the "barrier frame" logic) we may increase
    # the number of frames we use for prediction.
    # JT: note that this non-barrier-frame logic is not routinely used any more,
    # and could be improved a bit. I will leave all this for now, though, because I anticipate
    # it being replaced with better forward-prediction algorithms fairly soon anyway.
    frameInterval = 1.0 / pog_settings["framerate"]
    if allowedToExtendNumberOfFittedPoints and time_to_wait_seconds > (
        pog_settings["extrapolationFactor"] * framesForFit * frameInterval
    ):
        extendedFramesForFit = framesForFit * 2
        if (
            extendedFramesForFit <= frameHistory.shape[0]
            and extendedFramesForFit <= pog_settings["maxFramesForFit"]
        ):
            logger.info("Increasing number of frames to use")
            #  Recurse, using a larger number of frames, to obtain an improved predicted time
            # (Note that if we get to this code branch, fitBackToBarrier will in fact definitely be False)
            time_to_wait_seconds, radsPerSec = predict_trigger_wait(
                frame_history, pog_settings, fitBackToBarrier=fitBackToBarrier, framesForFit=extendedFramesForFit
            )

    # Return our prediction
    return time_to_wait_seconds, radsPerSec

def determine_barrier_frame_lookup(pog_settings):
    """ Identifies which past frame phases to use for forward-prediction of heart phase,
        depending on what our current frame phase is.
        This is empirical code replicated from the spim-interface project.
        The concept of the 'barrier frame' is used: we do not fit backwards past this
        barrier frame. The barrier frame is identified empirically as a point in the
        reference sequence "between" heartbeats, where we may we see increased variability
        (variable pause between beats). Our forward-prediction is more reliable if
        we do not attempt to extend our linear fit backwards past that point
        of high variability in the heart cycle.
        """
    # JT TODO: this function is called when we acquire a new set of reference frames,
    # but it is not updated if the parameters such as barrierFrame, min/maxFramesForFit
    # are later altered by the user. It should be...
    
    # frames to consider based on reference point and no padding
    numToUseNoPadding = list(
        range(pog_settings["referenceFrameCount"] - (2 * numExtraRefFrames))
    )
    numToUseNoPadding = np.asarray(
        numToUseNoPadding[-int(pog_settings["barrierFrame"] - 3) :]
        + numToUseNoPadding[: -int(pog_settings["barrierFrame"] - 3)]
    )

    # Account for padding by setting extra frames equal to last/first unpadded number
    numToUsePadding = numToUseNoPadding[-1] * np.ones(pog_settings["referenceFrameCount"])
    numToUsePadding[
        numExtraRefFrames : pog_settings["referenceFrameCount"] - numExtraRefFrames
    ] = numToUseNoPadding

    # Consider min and max number of frames to use, as determined by our settings parameters.
    # This overrides any barrier frame considerations.
    numToUsePadding = np.maximum(
        numToUsePadding,
        pog_settings["minFramesForFit"] * np.ones(pog_settings["referenceFrameCount"]),
    )
    numToUsePadding = np.minimum(
        numToUsePadding,
        pog_settings["maxFramesForFit"] * np.ones(pog_settings["referenceFrameCount"]),
    )

    return numToUsePadding

def pick_target_and_barrier_frames(reference_frames, reference_period):
    """ Function to automatically identify a stable target phase and barrier frame.
        Looks through 'reference_frames' to identify a consistent point in the
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
        consistent point in the cycle. For side-on orientations (as seen by
        the brightfield camera) of the fish, this does a reasonable job of
        identifying a point in the cycle that gives a fairly good sync.
        For belly-on orientations (as seen by the brightfield camera), it
        does not always work so well, and manual intervention is more likely
        to be needed to pick an appropriate target sync phase.

        Parameters:
            reference_frames    array-like  3D (t by x by y) frame pixel data for our reference period
            reference_period    float       Period (with sub-frame accuracy) associated with reference_frames
    """

    # First compare each frame in our list with the previous one
    # Note that this code assumes numExtraRefFrames>0 (which it certainly should be!)
    deltas_without_padding = np.zeros(
        (len(reference_frames) - 2 * numExtraRefFrames), dtype=np.int64,
    )
    for i in np.arange(len(reference_frames) - 2 * numExtraRefFrames,):
        deltas_without_padding[i] = jps.sad_correlation(
            reference_frames[i + numExtraRefFrames],
            reference_frames[i + numExtraRefFrames + 1],
        )

    min_pos_without_padding = np.argmin(deltas_without_padding)
    max_pos_without_padding = np.argmax(deltas_without_padding)
    min_delta = deltas_without_padding.min()
    max_delta = deltas_without_padding.max()
    target_frame_without_padding = 0

    # The greatest diff (most rapid change) tends to come soon after the region of minimum change. The greatest diff tends to be a sharp peak (whereas, almost by definition, the minimum change is broader) We use the greatest diff as our "universal"(ish) reference point, but default to a phase 1/3 of a period after it, so that we get a good clear run-up to it with well-defined phases in our history.

    # Use v-fitting to find a sub-frame estimate of the maximum
    if (max_pos_without_padding <= 0) or (max_pos_without_padding == deltas_without_padding.size - 1):
        # It looks as if the best position is right at the start or end of the dataset. Presumably due to a slightly glitch it's possible that the true minimum lies just outside the dataset. However, in pathological datasets there could be no minimum in easy reach at all, so we've got to give up at some point. Therefore, if we hit this condition, we just decide the best offset is 0.0. In sensible cases, that will be very close to optimum. In messy cases, this entire function is going to do unpredictable things anyway, so who cares!
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
        target_frame_without_padding + (reference_period / 3.0)
    ) % reference_period

    # We also identify a point soon after the region of minimum change ends. We look for the diffs to rise past their midpoint between min and max (which is expected to happen soon). We should try not to fit our history back beyond that, in order to avoid trying to fit to the (unpredictable) refractory period of the heart
    barrier_frame_without_padding = min_pos_without_padding
    while (
        deltas_without_padding[barrier_frame_without_padding]
        < (min_delta + max_delta) / 2
    ):
        barrier_frame_without_padding = (barrier_frame_without_padding + 1) % int(reference_period)
        if barrier_frame_without_padding == min_pos_without_padding:
            # This means I have done a complete loop and found no delta less than the average of the min and max deltas
            # Probably means my frames are all empty so my diffs are all zeros
            logger.warning(
                "Search for a barrier frame has failed: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}",
                min_pos_without_padding,
                min_delta,
                max_pos_without_padding,
                max_delta,
                target_frame_without_padding,
                barrier_frame_without_padding,
                reference_period,
                numExtraRefFrames,
            )
            break

    # Return new value for target_frame and barrier_frame
    logger.success(
        "Updating parameters with new target frame of {0} and barrier frame of {1} (both without padding).",
        target_frame_without_padding,
        barrier_frame_without_padding,
    )
    return target_frame_without_padding, barrier_frame_without_padding


def decide_whether_to_trigger(timestamp, timeToWaitInSeconds, pog_settings, heartRateRadsPerSec):
    """ Potentially schedules a synchronization trigger for the fluorescence camera,
        based on the caller-supplied candidate trigger time described by timeToWaitInSeconds.
        We will do this if the trigger is due fairly soon in the future,
        and we are not confident we will have time to make an updated prediction
        based on the next incoming frame from the brightfield camera.
        
        Parameters:
            timestamp               float   Time associated with current frame (seconds)
            timeToWaitInSeconds     float   Time delay before trigger would need to be sent.
            pog_settings            dict    Parameters controlling the sync algorithms
            heartRateRadsPerSec     float   Heart rate in radians per sec
        Returns:
            timeToWaitInSeconds     float   Time delay before trigger would need to be sent.
                                             Note that this return value may be modified from its input value (see code below).
            sendIt                  int     Nonzero indicates that a trigger for the fluorescence camera should be scheduled now,
                                             for a time timeToWaitInSeconds into the future.
        """
    sendIt = 0
    # 'framerateFactor' (in units of frames) is an empirical constant affecting the logic below.
    # Its value should ideally depend on the actual observed variability in the time estimates
    # as successive frame data is received
    framerateFactor = 1.6  # in frames

    logger.debug(
        "Time to wait: {0} s; with latency: {1} s;",
        timeToWaitInSeconds,
        pog_settings["prediction_latency_s"],
    )

    # The pog_settings parameter 'prediction_latency_s' represents how much time we *expect* to need
    # between scheduling a trigger and actually being able to send it.
    # That influences whether we commit to this trigger time, or wait for an updated prediction based on the next brightfield frame due to arrive soon
    if timeToWaitInSeconds < pog_settings["prediction_latency_s"]:
        logger.info(
            "Trigger due very soon, but if haven't already sent one this period then we may as well give it a shot..."
        )
        # JT TODO: does this "lastSent" logic make sense? At a glance, it looks like there should be a factor of 0.5 or something,
        # as I think we might *just* fail this test on the next heartbeat...?
        if pog_settings["lastSent"] < timestamp - (
            pog_settings["reference_period"] / pog_settings["framerate"]
        ):
            # Haven't sent a trigger on this heartbeat. Give it a go and cross our fingers we schedule it in time
            logger.success("Trigger will be sent")
            sendIt = 1
        else:
            # We have already sent a trigger on this heartbeat, so we consider that we are now making predictions for the *next* cycle.
            logger.info(
                "Trigger already sent recently. Will not send another - extending the prediction to the next cycle."
            )
            timeToWaitInSeconds += 2*np.pi / heartRateRadsPerSec
    elif (timeToWaitInSeconds - (framerateFactor / pog_settings["framerate"])) < pog_settings[
        "prediction_latency_s"
    ]:
        # We don't expect to have time to wait for an updated prediction... so schedule the trigger now!
        logger.success(
            "We don't expect to have time to wait for an updated prediction... so trigger scheduled now!"
        )
        sendIt = 2
    else:
        # We expect to have time to wait for an updated prediction, so we do nothing for now.
        pass

    if sendIt > 0 and pog_settings["lastSent"] > (
        timestamp - ((pog_settings["reference_period"] / pog_settings["framerate"]) / 2)
    ):
        # If we've done any triggering in the last half a cycle, don't trigger again.
        # This is quite different from JTs approach,
        # where he keeps track of which cycle we last triggered on.
        # JT note: the reason for my approach is because I may want to send multiple triggers at different heart phases
        # Future updates to this code can incorporate that concept...
        logger.info(
            "Trigger type {0} at {1}\tDROPPED", sendIt, timestamp + timeToWaitInSeconds
        )
        sendIt = 0
    elif sendIt > 0:
        logger.success("Trigger scheduled to be sent, updating `pog_settings['lastSent']` to {0}.", timestamp)
        # TODO: lastSent is perhaps misleading - this is about when we scheduled it, not when the electronic signal occurs.
        # In fact, I think the logic I actually use this variable for can still work if I change this to mostRecentTriggerTime...
        pog_settings["lastSent"] = timestamp

    return timeToWaitInSeconds, sendIt
