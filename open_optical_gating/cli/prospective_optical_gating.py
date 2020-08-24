"""Module for real-time phase matching of brightfield sequences.
These codes are equivalent to the Objective-C codes in spim-interface."""

# Python imports
import numpy as np

# Module imports
from loguru import logger
import j_py_sad_correlation as jps

# Local imports
import open_optical_gating.cli.parameters as parameters

# TODO: JT writes: numExtraRefFrames should really be a global constant, not a parameter in settings.
# Really the only reason that parameter exists at all in the C code is to self-document all the +-2 arithmetic that would otherwise appear.
# In the C code it is declared as a const int.


def update_drift(frame0, bestMatch0, settings):
    """ Updates the 'settings' dictionary to reflect our latest estimate of the sample drift.
        We do this by trying variations on the relative shift between frame0 and the best-matching frame in the reference sequence.
        
        Parameters:
            frame0      array-like      2D frame pixel data for our most recently-received frame
            bestMatch0  array-like      2D frame pixel data for the best match within our reference sequence
            settings    dict            Parameters controlling the sync algorithms
        Returns:
            updated settings dictionary
        """
    # frame0 and bestMatch0 must be numpy arrays of the same size
    assert frame0.shape == bestMatch0.shape

    # Start with the existing drift parameters in the settings dictionary
    dx, dy = settings["drift"]

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

    # Build up a set of frames each representing a window into frame0 with slightly different drift offsets
    frames = np.zeros(
        [len(candidateShifts), bestMatch.shape[0], bestMatch.shape[1]],
        dtype=frame0.dtype,
    )
    counter = 0
    for shft in candidateShifts:
        dxp = dx + shft[0]
        dyp = dy + shft[1]

        # Adjust for drift and shift
        rectF = np.copy(rect)
        rectF[0] -= dxp
        rectF[1] -= dxp
        rectF[2] -= dyp
        rectF[3] -= dyp

        frames[counter, :, :] = frame0[rectF[0] : rectF[1], rectF[2] : rectF[3]]
        counter = counter + 1

    # Compare all these candidate shifted images against the matching reference frame, and find the best-matching shift
    sad = jps.sad_with_references(bestMatch, frames)
    best = np.argmin(sad)

    settings["drift"][0] = dx + candidateShifts[best][0]
    settings["drift"][1] = dy + candidateShifts[best][1]

    return settings


def subframe_fitting(diffs, settings):
    """ Identify the location of the minimum value (to sub-frame accuracy) within a list of diffs (including padding frames).
        The aim is to find the position of the minimum within the "main" frames,
        but we may have to extend to the padding frames if the minimum is right at one end of the "main" frames.
        
        Parameters:
            diffs       list/array-like  Sequence of recently-received frame pixel arrays (in chronological order)
            settings    dict             Parameters controlling the sync algorithms
        Returns:
            float coordinate for location of minimum in 'diffs' (including padding frames)
    """
    # Search for lowest value within the "main" frames.
    # Note that this code assumes "numExtraRefFrames">0 (which it certainly should be!)
    bestScorePos = np.argmin(
        diffs[settings["numExtraRefFrames"] : -settings["numExtraRefFrames"]]
    )
    bestScorePos = bestScorePos + settings["numExtraRefFrames"]

    # Sub-pixel fitting
    if diffs[bestScorePos - 1] < diffs[bestScorePos]:  # If no V is found
        # The minimum is right at one end of our reference sequence.
        # If we get here then there is no actual "v" minimum to be found within the range of our
        # main reference frames. It is probably to be found just outside, and we could
        # potentially attempt to fit it anyway. However this should be an unusual occurrence,
        # and an assumption of a phase of 0 shouldn't be too far from the truth,
        # so to keep things simple and robust I'm going to leave it at this for now.
        # We just return a reference phase that corresponds to a phase of 0
        # JT TODO: surely we should also test for this condition at the upper end of the reference sequence,
        # as well as the lower end (which is what we test here)
        thisFrameReferencePos = settings["numExtraRefFrames"]
        logger.warning("No minimum found - defaulting to phase=0")
    else:
        # A minimum exists - do sub-frame interpolation on it
        interpolatedCorrection, _ = v_fitting(
            diffs[bestScorePos - 1], diffs[bestScorePos], diffs[bestScorePos + 1]
        )
        thisFrameReferencePos = bestScorePos + interpolatedCorrection

    return thisFrameReferencePos


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


def phase_matching(frame, reference_frames, settings=None):
    """Phase match a new frame based on a reference period.
        
        Parameters:
            frame               array-like      2D frame pixel data for our most recently-received frame
            reference_frames    array-like      3D (t by x by y) frame pixel data for our reference period
            settings            dict            Parameters controlling the sync algorithms
        Returns:
            phase               float           phase matching results
            SADs                ndarray         1D sum of absolute differences between frame and each reference_frames[t,...]
            settings            dict            updated parameters controlling the sync algorithms
    """
    if settings == None:
        logger.warning("No settings provided. Using sensible defaults.")
        settings = parameters.initialise()

    dx, dy = settings["drift"]

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
    reference_frames_cropped = reference_frames[:, rect[0] : rect[1], rect[2] : rect[3]]

    # Calculate SADs
    logger.trace(
        "Reference frame dtypes: {0} and {1}", frame.dtype, reference_frames.dtype
    )
    logger.trace(
        "Reference frame shapes: {0} and {1}", frame.shape, reference_frames.shape
    )
    logger.trace(
        "Reference frames range over {0} to {1} and {2} to {3}.",
        frame_cropped.min(),
        frame_cropped.max(),
        reference_frames_cropped.min(),
        reference_frames_cropped.max(),
    )
    # DEV NOTE: these two .astypes aren't ideal (especially as the data is already 8-bit)
    # However they were needed when running the picam live to stop jps falling down on types
    # This is also why the above three traces seem a bit overkill
    SADs = jps.sad_with_references(
        frame_cropped.astype("uint8"), reference_frames_cropped.astype("uint8")
    )
    logger.trace(SADs)

    # Identify best match between 'frame' and the reference frame sequence
    phase = subframe_fitting(SADs, settings)
    logger.debug("Found frame phase to be {0}", phase)

    # Update current drift estimate in the settings dictionary
    settings = update_drift(frame, reference_frames[np.argmin(SADs)], settings)
    logger.info(
        "Drift correction updated to ({0},{1})",
        settings["drift"][0],
        settings["drift"][1],
    )

    # Note: still includes padding frames (on purpose)   [TODO: JT writes: what does!? I presume the SAD array. Can the comment explain *why* this is done on purpose?]
    return (phase, SADs, settings)


def predict_trigger_wait(frame_history, settings, fitBackToBarrier=True):
    """ Predict how long we need to wait until the heart is at the target phase we are triggering to.
        
        Parameters:
            frame_history           array-like  Nx3 array of [timestamp in seconds, phase, argmin(SAD)]
                                                 Phase (i.e. frame_history[:,1]) should be cumulative (i.e. phase-UNwrapped) phase in radians
            settings                dict        Parameters controlling the sync algorithms
                                                 targetSyncPhase is expected to be in [0,2pi]
            fitBackToBarrier        bool        Should we use the "barrier frame" logic? (see determine_barrier_frames)
        Returns:
            Time delay (or phase delay) before trigger would need to be sent in seconds.
        """

    if frame_history.shape[0] < settings["minFramesForFit"]:
        logger.debug("Fit failed due to too few frames...")
        return -1

    # Deal with the barrier frame logic (if fitBackToBarrier is True):
    # Rather than fitting to a number of frames that depends on how far forward we are predicting, fit to a number that depends on where in the cycle we are. We try not to fit to the refractory period unless there really is no other data. The intention of this is to fit to as much data as possible but only in the parts of the cycle where the phase progression is highly predictable and linear with time.
    if fitBackToBarrier:
        allowedToExtendNumberOfFittedPoints = False
        framesForFit = min(
            settings["frameToUseArray"][int(frame_history[-1, 2])],
            frame_history.shape[0],
        )
        logger.debug("Consider {0} past frames for prediction;", framesForFit)
    else:
        framesForFit = settings["minFramesForFit"]
        allowedToExtendNumberOfFittedPoints = True
    pastPhases = frame_history[-int(framesForFit) : :, :]

    # Perform a linear fit to the past phases. We will use this for our forward-prediction
    radsPerSec, alpha = np.polyfit(pastPhases[:, 0], pastPhases[:, 1], 1)

    logger.trace(pastPhases[:, 0])
    logger.trace(pastPhases[:, 1])

    logger.info("Linear fit with intersect {0} and gradient {1}", alpha, radsPerSec)
    if radsPerSec < 0:
        logger.warning(
            "Linear fit to unwrapped phases is negative! This is a problem (fakeNews)."
        )
    elif radsPerSec == 0:
        logger.warning(
            "Linear fit to unwrapped phases is zero! This will be a problem for prediction (divByZero)."
        )

    # Use our linear fit to get a 'fitted' unwraped phase for the latest frame
    # This should not rescue cases where, for some reason, the image-based
    # phase matching is erroneous.
    thisFramePhase = alpha + frame_history[-1, 0] * radsPerSec
    # Count how many total periods we have seen
    multiPhaseCounter = thisFramePhase // (2 * np.pi)
    # Determine how much of a cardiac cycle we have to wait till our target phase
    phaseToWait = (
        settings["targetSyncPhase"] + (multiPhaseCounter * 2 * np.pi) - thisFramePhase
    )
    # c.f. function triggerAnticipationProcessing in SyncAnalyzer.mm
    # essentially this fixes for small backtracks in phase due to SAD imperfections
    while phaseToWait < 0:
        phaseToWait += 2 * np.pi

    timeToWaitInSecs = phaseToWait / radsPerSec
    timeToWaitInSecs = max(timeToWaitInSecs, 0.0)

    logger.info(
        "Current time: {0};\tTime to wait: {1};",
        frame_history[-1, 0],
        timeToWaitInSecs,
    )
    logger.debug(
        "Current phase: {0};\tPhase to wait: {1};", thisFramePhase, phaseToWait,
    )
    logger.debug(
        "Target phase:{0};\tPredicted phase:{1};",
        settings["targetSyncPhase"] + (multiPhaseCounter * 2 * np.pi),
        thisFramePhase + phaseToWait,
    )

    # Fixes sync error due to targetSyncPhase being 2pi greater than target phase (1e-3 is for floating point errors)
    if (
        thisFramePhase
        + phaseToWait
        - settings["targetSyncPhase"]
        - (multiPhaseCounter * 2 * np.pi)
        > 2 * np.pi + 1e-3
    ):
        logger.warning(
            "Phase discrepency, trigger aborted. At {0} with wait {1} for target {2} [{3}]",
            thisFramePhase % (2 * np.pi),
            phaseToWait,
            settings["targetSyncPhase"],
            thisFramePhase
            + phaseToWait
            - settings["targetSyncPhase"]
            - (multiPhaseCounter * 2 * np.pi),
        )
        timeToWaitInSecs = 0.0

    # This logic catches cases where we are predicting a long way into the future using only a small number of datapoints.
    # That is likely to be error-prone, so (unless using the "barrier frame" logic) we may increase
    # the number of frames we use for prediction.
    frameInterval = 1.0 / settings["framerate"]
    if allowedToExtendNumberOfFittedPoints and timeToWaitInSecs > (
        settings["extrapolationFactor"] * settings["minFramesForFit"] * frameInterval
    ):
        # TODO: JT writes: this approach of editing settings[minFramesForFit] and then changing it back again feels really messy to me.
        # We should hold off changing it for now, though, because I think we may want to refactor how settings is used
        # Once that is complete, a better solution here may present itself naturally.
        settings["minFramesForFit"] *= 2
        if (
            settings["minFramesForFit"] <= pastPhases.shape[0]
            and settings["minFramesForFit"] <= settings["maxFramesForFit"]
        ):
            logger.info("Increasing number of frames to use")
            #  Recurse, using a larger number of frames, to obtain an improved predicted time
            timeToWaitInSecs = predict_trigger_wait(
                frame_history, settings, fitBackToBarrier=False
            )
        settings["minFramesForFit"] = settings["minFramesForFit"] // 2

    # Return our prediction
    return timeToWaitInSecs


def determine_barrier_frames(settings):
    """ Identifies which past frame phases to use for forward-prediction of heart phase,
        depending on what our current frame phase is.
        This is empirical code replicated from the spim-interface project.
        The concept of the 'barrier frame' is used: we do not fit backwards past this
        barrier frame. The barrier frame is identified empirically as a point in the
        reference sequence "between" heartbeats, where we may we see increased variability
        (variable pause between beats). Our forward-prediction is more reliable if
        we do not attempt to extend our linear fit backwards past that point 
        of high variability in the heart cycle.
        
        Parameters:
            settings    dict            Parameters controlling the sync algorithms
        Returns:
            updated settings dictionary
        """
    # frames to consider based on reference point and no padding
    numToUseNoPadding = list(
        range(settings["referenceFrameCount"] - (2 * settings["numExtraRefFrames"]))
    )
    numToUseNoPadding = np.asarray(
        numToUseNoPadding[-int(settings["barrierFrame"] - 3) :]
        + numToUseNoPadding[: -int(settings["barrierFrame"] - 3)]
    )

    # Account for padding by setting extra frames equal to last/first unpadded number
    numToUsePadding = numToUseNoPadding[-1] * np.ones(settings["referenceFrameCount"])
    numToUsePadding[
        settings["numExtraRefFrames"] : (
            settings["referenceFrameCount"] - settings["numExtraRefFrames"]
        )
    ] = numToUseNoPadding

    # Consider min and max number of frames to use, as configured in the settings.
    # This overrides any barrier frame considerations.
    numToUsePadding = np.maximum(
        numToUsePadding,
        settings["minFramesForFit"] * np.ones(settings["referenceFrameCount"]),
    )
    numToUsePadding = np.minimum(
        numToUsePadding,
        settings["maxFramesForFit"] * np.ones(settings["referenceFrameCount"]),
    )

    # Update settings and return
    settings["frameToUseArray"] = numToUsePadding
    return settings


def pick_target_and_barrier_frames(reference_frames, settings):
    """Function to automatically identify a stable target phase and barrier frame.
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
            settings            dict        parameters controlling the sync algorithms
        Returns:
            settings            dict        updated settings
    """

    # First compare each frame in our list with the previous one
    # Note that this code assumes "numExtraRefFrames">0 (which it certainly should be!)
    deltas_without_padding = np.zeros(
        (reference_frames.shape[0] - 2 * settings["numExtraRefFrames"]), dtype=np.int64,
    )
    for i in np.arange(reference_frames.shape[0] - 2 * settings["numExtraRefFrames"],):
        deltas_without_padding[i] = jps.sad_correlation(
            reference_frames[i + settings["numExtraRefFrames"], :, :],
            reference_frames[i + settings["numExtraRefFrames"] + 1, :, :],
        )

    min_pos_without_padding = np.argmin(deltas_without_padding)
    max_pos_without_padding = np.argmax(deltas_without_padding)
    min_delta = deltas_without_padding.min()
    max_delta = deltas_without_padding.max()
    target_frame_without_padding = 0

    # The greatest diff (most rapid change) tends to come soon after the region of minimum change. The greatest diff tends to be a sharp peak (whereas, almost by definition, the minimum change is broader) We use the greatest diff as our "universal"(ish) reference point, but default to a phase 1/3 of a period after it, so that we get a good clear run-up to it with well-defined phases in our history.

    # Use v-fitting to find a sub-frame estimate of the maximum
    if max_pos_without_padding <= 0:
        # It looks as if the best position is right at the start of the dataset. Presumably due to a slightly glitch it's possible that the true minimum lies just outside the dataset. However, in pathological datasets there could be no minimum in easy reach at all, so we've got to give up at some point. Therefore, if we hit this condition, we just decide the best offset is 0.0. In sensible cases, that will be very close to optimum. In messy cases, this entire function is going to do unpredictable things anyway, so who cares!
        target_frame_without_padding = 0
    else:
        target_frame_without_padding = max_pos_without_padding + v_fitting(
            -deltas_without_padding[max_pos_without_padding - 1],
            -deltas_without_padding[max_pos_without_padding],
            -deltas_without_padding[max_pos_without_padding + 1],
        )

    # Now shift that forwards by 1/3 of a period but wrap if this is in the next beat
    target_frame_without_padding = (
        target_frame_without_padding + (settings["reference_period"] / 3.0)
    ) % settings["reference_period"]

    # We also identify a point soon after the region of minimum change ends. We look for the diffs to rise past their midpoint between min and max (which is expected to happen soon). We should try not to fit our history back beyond that, in order to avoid trying to fit to the (unpredictable) refractory period of the heart
    barrier_frame_without_padding = min_pos_without_padding
    while (
        deltas_without_padding[barrier_frame_without_padding]
        < (min_delta + max_delta) / 2
    ):
        barrier_frame_without_padding = (barrier_frame_without_padding + 1) % int(
            settings["reference_period"]
        )
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
                settings["reference_period"],
                settings["numExtraRefFrames"],
            )
            break

    # Update settings with new target_phase and barrier_frame
    logger.success(
        "Updating parameters with new target frame of {0} and barrier frame of {1} (both without padding).",
        target_frame_without_padding,
        barrier_frame_without_padding,
    )
    settings = parameters.update(
        settings,
        referenceFrame=target_frame_without_padding,
        barrierFrame=barrier_frame_without_padding,
    )

    return settings


def decide_trigger(timestamp, timeToWaitInSeconds, settings):
    """ Potentially schedules a synchronization trigger for the fluorescence camera.
        We will do this if the trigger is due fairly soon in the future,
        and we are not confident we will have time to make an updated prediction
        based on the next incoming frame from the brightfield camera.
        
        Parameters:
            timestamp               float   Time associated with current frame (seconds)
            timeToWaitInSeconds     float   Time delay before trigger would need to be sent.
            settings                dict    Parameters controlling the sync algorithms
        Returns:
            timeToWaitInSeconds     float   Time delay before trigger would need to be sent.
                                             Note that this returned  may be modified from its input value (see code below).
            sendIt                  int     Nonzero indicates that a trigger for the fluorescence camera should be scheduled now,
                                             for a time timeToWaitInSeconds into the future.
            settings                dict    Updated settings dictionary
        """
    sendIt = 0
    # 'framerateFactor' (in units of frames) is an emprical constant affecting the logic below.
    # Its value should ideally depend on the actual observed variability in the time estimates
    # as successive frame data is received
    framerateFactor = 1.6  # in frames

    logger.debug(
        "Time to wait: {0} s; with latency: {1} s;",
        timeToWaitInSeconds,
        settings["prediction_latency_s"],
    )

    # The settings parameter 'prediction_latency_s' represents how much time we *expect* to need
    # between scheduling a trigger and actually being able to send it.
    # That influences whether we commit to this trigger time, or wait for an updated prediction based on the next brightfield frame due to arrive soon
    if timeToWaitInSeconds < settings["prediction_latency_s"]:
        logger.info(
            "Trigger due very soon, but if haven't already sent one this period then we may as well give it a shot..."
        )
        if settings["lastSent"] < timestamp - (
            settings["reference_period"] / settings["framerate"]
        ):
            # Haven't sent a trigger on this heartbeat. Give it a go and cross our fingers we schedule it in time
            logger.success("Trigger will be sent")
            sendIt = 1
        else:
            # We have already sent a trigger on this heartbeat, so we consider that we are now making predictions for the *next* cycle.
            # The value added to timeToWaitInSeconds is a rather crude method to extend the prediction to the next cycle (it assumes
            # that the heart rate is the same as with the reference sequence). However, this prediction is not crucial because
            # clearly we will get much better estimates nearer the time. Really we are just returning this as something vaguely sensible,
            # for cosmetic reasons.
            logger.info(
                "Trigger already sent recently. Will not send another - extending the prediction to the next cycle."
            )
            timeToWaitInSeconds += settings["reference_period"] / settings["framerate"]
    elif (timeToWaitInSeconds - (framerateFactor / settings["framerate"])) < settings[
        "prediction_latency_s"
    ]:
        # We don't expect to have time to wait for an updated prediction... so schuedule the trigger now!
        logger.success(
            "We don't expect to have time to wait for an updated prediction... so trigger scheduled now!"
        )
        sendIt = 2
    else:
        # We expect to have time to wait for an updated prediction, so we do nothing for now.
        pass

    if sendIt > 0 and settings["lastSent"] > (
        timestamp - ((settings["reference_period"] / settings["framerate"]) / 2)
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
        logger.success("Trigger scheduled to be sent, updating `settings['lastSent']`.")
        settings["lastSent"] = timestamp

    return timeToWaitInSeconds, sendIt, settings
