"""Module for real-time phase matching of brightfield sequences.
These codes are equivalent to the Objective-C codes in spim-interface."""

# Python imports
import numpy as np

# Module imports
from loguru import logger
import j_py_sad_correlation as jps

# Local imports
import open_optical_gating.cli.parameters as parameters


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
    bestScore = diffs[settings["numExtraRefFrames"]]
    bestScorePos = settings["numExtraRefFrames"]

    # Search for lowest value within the "main" frames.
    # TODO: not a real performance issue, but this can easily be rewritten loop-free using np.argmax etc.
    for i in range(
        settings["numExtraRefFrames"] + 1, len(diffs) - settings["numExtraRefFrames"]
    ):
        # If new lower V
        if (
            diffs[i] < bestScore
            and diffs[i - 1] >= diffs[i]
            and diffs[i + 1] >= diffs[i]
        ):
            bestScore = diffs[i]
            bestScorePos = i

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


def phase_matching(frame0, referenceFrames0, settings=None):
    # assumes frame is a numpy array and referenceFrames is a dictionary of {phase value: numpy array}
    # TODO: JT writes: check the above description of referenceFrames -  I have no idea what this means, and I don’t think it is correct either. referenceFrames0 seems to just behave like a 3D array?
    # TODO: JT writes: Supply a proper function header [CJN: and better variable names]. That should also explain what xxx0 represents [i.e. uncropped], vs xxx [cropped].

    logger.debug(
        "Reference frame types: {0} and {1}", type(frame0), type(referenceFrames0)
    )
    logger.debug(
        "Reference frame dtypes: {0} and {1}", frame0.dtype, referenceFrames0.dtype
    )
    logger.debug(
        "Reference frame shapes: {0} and {1}", frame0.shape, referenceFrames0.shape
    )
    if settings == None:
        logger.warning("No settings provided. Using sensible deafults.")
        settings = parameters.initialise()

    dx, dy = settings["drift"]

    # Apply drift correction, identifying a crop rect for the frame and/or reference frames,
    # representing the area intersection between them once drift is accounted for.
    logger.info("Applying drift correction of ({0},{1})", dx, dy)
    rectF = [0, frame0.shape[0], 0, frame0.shape[1]]  # X1,X2,Y1,Y2
    rect = [
        0,
        referenceFrames0[0].shape[0],
        0,
        referenceFrames0[0].shape[1],
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

    frame = frame0[rectF[0] : rectF[1], rectF[2] : rectF[3]]
    referenceFrames = referenceFrames0[:, rect[0] : rect[1], rect[2] : rect[3]]
    # if plot:
    #    a12 = f1.add_subplot(122)
    #    a12.imshow(frame)
    #    plt.show()

    # Calculate SADs
    logger.trace(
        "Reference frame dtypes: {0} and {1}", frame0.dtype, referenceFrames0.dtype
    )
    logger.trace(
        "Reference frame shapes: {0} and {1}", frame0.shape, referenceFrames0.shape
    )
    logger.trace(
        "Reference frames range over {0} to {1} and {2} to {3}.",
        frame.min(),
        frame.max(),
        referenceFrames.min(),
        referenceFrames.max(),
    )
    SADs = jps.sad_with_references(
        frame.astype("uint8"), referenceFrames.astype("uint8")
    )

    logger.trace(SADs)
    # if plot:
    #    f2 = plt.figure()
    #    a21 = f2.add_axes([0, 0, 1, 1])
    #    a21.plot(range(len(SADs)), SADs)
    #    plt.show()

    # Identify best match between 'frame' and the reference frame sequence
    phase = subframe_fitting(SADs, settings)
    logger.debug("Found frame phase to be {0}", phase)

    # Update current drift estimate in the settings dictionary
    settings = update_drift(frame0, referenceFrames0[np.argmin(SADs)], settings)
    logger.info(
        "Drift correction updated to ({0},{1})",
        settings["drift"][0],
        settings["drift"][1],
    )

    # Note: still includes padding frames (on purpose)   [TODO: JT writes: what does!? I presume the SAD array. Can the comment explain *why* this is done on purpose?]
    return (phase, SADs, settings)


def predict_trigger_wait(
    frame_history, settings, fitBackToBarrier=True
):
    """ Predict how long we need to wait until the heart is at the target phase we are triggering to.
        
        Parameters:
            frame_history           array-like  Nx3 array of [timestamp, phase, argmin(SAD)]
                                                 Phase (i.e. frame_history[:,1]) should be cumulative (i.e. phase-UNwrapped) phase in radians
            settings                dict        Parameters controlling the sync algorithms
                                                 targetSyncPhase should is expected to be in [0,2pi]
            fitBackToBarrier        bool        Should we use the "barrier frame" logic? (see determine_barrier_frames)
        Returns:
            Time delay (or phase delay) before trigger would need to be sent.
        """

    # TODO: JT writes: I removed the +1 from the following "if" test because I can’t see any logical purpose for it. Does that seem correct to you?
    if frame_history.shape[0] < settings["minFramesForFit"]:
        logger.debug("Fit failed due to too few frames...")
        return -1

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
    # === Perform a linear fit to the past phases. We will use this for our forward-prediction ===
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

    # === Turn our linear fit into a future prediction ===
    thisFramePhase = (
        alpha + frame_history[-1, 0] * radsPerSec
    )  # Use the *fitted* phase for this current frame
    multiPhaseCounter = thisFramePhase // (2 * np.pi)
    phaseToWait = (
        settings["targetSyncPhase"] + (multiPhaseCounter * 2 * np.pi) - thisFramePhase
    )
    # c.f. function triggerAnticipationProcessing in SyncAnalyzer.mm
    # essentially this fixes for small backtracks in phase due to SAD imperfections
    while (
        phaseToWait < 0
    ):  # this used to be -np.pi       # TODO: JT writes: who added this comment? Does it serve any purpose? <0 seems like the right test to me (and is what I used)
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

    # Fixes sync error due to targetSyncPhase being 2pi greater than target phase
    # TODO check with JTs system
    # TODO: JT writes: I have no idea what the above comment is about. Can this be clarified, or investigated?
    if (
        thisFramePhase
        + phaseToWait
        - settings["targetSyncPhase"]
        - multiPhaseCounter * 2 * np.pi
        > 0.1
    ):
        logger.warning(
            "Phase discrepency, trigger aborted. At {0} with wait {1} for target {2} [{3}]",
            thisFramePhase % (2 * np.pi),
            phaseToWait,
            settings["targetSyncPhase"],
            thisFramePhase - (multiPhaseCounter * 2 * np.pi),
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
        # TODO: JT writes: BUG: The recursive function call is also using completely the wrong parameters!!
        # (This may not have been hit in testing because you are probably using barrier frames, which prevent this code branch from running)
        settings["minFramesForFit"] *= 2
        if (
            settings["minFramesForFit"] <= pastPhases.shape[0]
            and settings["minFramesForFit"] <= settings["maxFramesForFit"]
        ):
            logger.info("Increasing number of frames to use")
            #  Recurse, using a larger number of frames, to obtain an improved predicted time
            timeToWaitInSecs = predict_trigger_wait(  # TODO: JT writes: this function call is passing completely the wrong parameters!!
                pastPhases,
                settings["targetSyncPhase"]
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
        settings["prediction_latency"],
    )

    # The settings parameter 'prediction_latency' represents how much time we *expect* to need
    # between scheduling a trigger and actually being able to send it.
    # That influences whether we commit to this trigger time, or wait for an updated prediction based on the next brightfield frame due to arrive soon
    if timeToWaitInSeconds < settings["prediction_latency"]:
        logger.info(
            "Trigger due very soon, but if haven't already sent one this period then we may as well give it a shot..."
        )
        if settings["lastSent"] < timestamp - (
            settings["referencePeriod"] / settings["framerate"]
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
            timeToWaitInSeconds += settings["referencePeriod"] / settings["framerate"]
    elif (timeToWaitInSeconds - (framerateFactor / settings["framerate"])) < settings[
        "prediction_latency"
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
        timestamp - ((settings["referencePeriod"] / settings["framerate"]) / 2)
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
