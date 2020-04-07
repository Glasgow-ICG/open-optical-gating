"""Module for real-time phase matching of brightfield sequences.
These codes are equivalent to the Objective-C codes in spim-interface."""

# Python imports
import numpy as np

# Module imports
from loguru import logger
import j_py_sad_correlation as jps

# Local imports
from . import parameters


def update_drift(frame0, bestMatch0, settings):
    # Assumes frame and bestMatch are numpy arrays of the same size

    dx = settings["drift"][0]
    dy = settings["drift"][1]

    # Default inset areas
    rect = [
        abs(dx) + 1,
        frame0.shape[0] - abs(dx) - 1,
        abs(dy) + 1,
        frame0.shape[1] - abs(dy) - 1,
    ]  # X1,X2,Y1,Y2
    bestMatch = bestMatch0[rect[0] : rect[1], rect[2] : rect[3]]

    shifts = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]

    frames = np.zeros([5, bestMatch.shape[0], bestMatch.shape[1]], dtype=frame0.dtype)

    counter = 0
    for shft in shifts:
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

    sad = jps.sad_with_references(bestMatch, frames)
    best = np.argmin(sad)

    settings["drift"][0] = dx + shifts[best][0]
    settings["drift"][1] = dy + shifts[best][1]

    return settings


def subframe_fitting(diffs, settings):
    # Find the sub-pixel position fit from a list of SADs
    # (including padding frames)
    # Initialise best and worst scores based on padding frames
    bestScore = diffs[settings["numExtraRefFrames"]]
    bestScorePos = settings["numExtraRefFrames"]

    # Search for lowest V
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
        thisFrameReferencePos = settings["numExtraRefFrames"]
    else:  # Sub-pixel fitting
        interpolatedCorrection = v_fitting(
            diffs[bestScorePos - 1], diffs[bestScorePos], diffs[bestScorePos + 1]
        )[0]
        thisFrameReferencePos = bestScorePos + interpolatedCorrection

    return thisFrameReferencePos


def v_fitting(y1, y2, y3):
    # Fit an even V to three points at x=-1, x=0 and x=+1
    if y1 > y3:
        x = 0.5 * (y1 - y3) / (y1 - y2)
        y = y2 - x * (y1 - y2)
    else:
        x = 0.5 * (y1 - y3) / (y3 - y2)
        y = y2 + x * (y3 - y2)

    return x, y


def phase_matching(frame0, referenceFrames0, settings=None):
    # assumes frame is a numpy array and referenceFrames is a dictionary of {phase value: numpy array}

    logger.trace(
        "Reference frame types: {0} and {1}", type(frame0), type(referenceFrames0)
    )
    logger.trace(
        "Reference frame dtypes: {0} and {1}", frame0.dtype, referenceFrames0.dtype
    )
    logger.trace(
        "Reference frame shapes: {0} and {1}", frame0.shape, referenceFrames0.shape
    )
    if settings == None:
        logger.warning("No settings provided. Using sensible deafults.")
        settings = parameters.initialise()

    dx = settings["drift"][0]
    dy = settings["drift"][1]

    # Apply shifts
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

    # Identify best match
    phase = subframe_fitting(SADs, settings)
    logger.debug("Found frame phase to be {0}", phase)

    # Update drift
    settings = update_drift(frame0, referenceFrames0[np.argmin(SADs)], settings)
    logger.info(
        "Drift correction updated to ({0},{1})",
        settings["drift"][0],
        settings["drift"][1],
    )

    # Note: still includes padding frames (on purpose)
    return (phase, SADs, settings)


def predict_trigger_wait(
    frameSummaryHistory, settings, fitBackToBarrier=True, output="seconds"
):
    # frameSummaryHistory is an nx3 array of [timestamp, phase, argmin(SAD)]
    # phase (i.e. frameSummaryHistory[:,1]) should be cumulative 2Pi phase
    # targetSyncPhase should be in [0,2pi]

    if frameSummaryHistory.shape[0] < settings["minFramesForFit"] + 1:
        logger.debug("Fit failed due to too few frames...")
        return -1
    if fitBackToBarrier:
        allowedToExtendNumberOfFittedPoints = False
        framesForFit = min(
            settings["frameToUseArray"][int(frameSummaryHistory[-1, 2])],
            frameSummaryHistory.shape[0],
        )
        logger.debug("Consider {0} past frames for prediction;", framesForFit)
    else:
        framesForFit = settings["minFramesForFit"]
        allowedToExtendNumberOfFittedPoints = True

    pastPhases0 = frameSummaryHistory[-int(framesForFit) :, :]

    radsPerSec, alpha = np.polyfit(pastPhases0[:, 0], pastPhases0[:, 1], 1)

    logger.info("Linear fit with intersect {0} and gradient {1}", alpha, radsPerSec)
    if radsPerSec < 0:
        logger.warning(
            "Linear fit to unwrapped phases is negative! This is a problem (fakeNews)."
        )
    elif radsPerSec == 0:
        logger.warning(
            "Linear fit to unwrapped phases is zero! This will be a problem for prediction (divByZero)."
        )

    thisFramePhase = alpha + frameSummaryHistory[-1, 0] * radsPerSec
    multiPhaseCounter = thisFramePhase // (2 * np.pi)
    phaseToWait = (
        settings["targetSyncPhase"] + (multiPhaseCounter * 2 * np.pi) - thisFramePhase
    )
    # c.f. lines 1798-1801 in SyncAnalyzer.mm
    # essentially this fixes for small drops in phase due to SAD errors
    while phaseToWait < 0:  # this used to be -np.pi
        phaseToWait += 2 * np.pi

    timeToWaitInSecs = phaseToWait / radsPerSec
    timeToWaitInSecs = max(timeToWaitInSecs, 0.0)

    logger.info(
        "Current time: {0};\tTime to wait: {1};",
        frameSummaryHistory[-1, 0],
        timeToWaitInSecs,
    )
    logger.debug(
        "Current phase: {0};\tPhase to wait: {1};\nTarget phase:{2};\tPredicted phase:{3};",
        thisFramePhase,
        phaseToWait,
        settings["targetSyncPhase"] + (multiPhaseCounter * 2 * np.pi),
        thisFramePhase + phaseToWait,
    )

    # Fixes sync error due to targetSyncPhase being 2pi greater than target phase
    if (
        thisFramePhase
        + phaseToWait
        - settings["targetSyncPhase"]
        - multiPhaseCounter * 2 * np.pi
        > 0.1
    ):
        logger.warning("Phase discrepency, trigger aborted.")
        timeToWaitInSecs = 0.0

    frameInterval = 1.0 / settings["framerate"]
    if allowedToExtendNumberOfFittedPoints and timeToWaitInSecs > (
        settings["extrapolationFactor"] * settings["minFramesForFit"] * frameInterval
    ):
        settings["minFramesForFit"] *= 2
        if (
            settings["minFramesForFit"] <= pastPhases.shape[0]
            and settings["minFramesForFit"] <= settings["maxFramesForFit"]
        ):
            logger.info("Increasing number of frames to use")
            timeToWaitInSecs = predict_trigger_wait(
                pastPhases, settings["targetSyncPhase"]
            )
        settings["minFramesForFit"] = settings["minFramesForFit"] // 2

    if output == "seconds":
        return timeToWaitInSecs
    elif output == "phases":
        return phaseToWait
    else:
        logger.critical("What are ye on mate!")
        return 0.0


def determine_barrier_frames(settings):
    # frames to consider based on reference point and no padding
    numToUseNoPadding = list(
        range(settings["referenceFrameCount"] - (2 * settings["numExtraRefFrames"]))
    )
    numToUseNoPadding = np.asarray(
        numToUseNoPadding[-int(settings["barrierFrame"] - 3) :]
        + numToUseNoPadding[: -int(settings["barrierFrame"] - 3)]
    )

    # consider padding by setting extra frames equal to last/first unpadded number
    numToUsePadding = numToUseNoPadding[-1] * np.ones(settings["referenceFrameCount"])
    numToUsePadding[
        settings["numExtraRefFrames"] : (
            settings["referenceFrameCount"] - settings["numExtraRefFrames"]
        )
    ] = numToUseNoPadding

    # consider min and max number of frames to use
    numToUsePadding = np.maximum(
        numToUsePadding,
        settings["minFramesForFit"] * np.ones(settings["referenceFrameCount"]),
    )
    numToUsePadding = np.minimum(
        numToUsePadding,
        settings["maxFramesForFit"] * np.ones(settings["referenceFrameCount"]),
    )

    # update settings
    settings["frameToUseArray"] = numToUsePadding
    return settings


def decide_trigger(timestamp, timeToWaitInSeconds, settings):
    sendIt = 0
    framerateFactor = 1.6  # in frames

    logger.trace(timeToWaitInSeconds, settings["predictionLatency"])

    if timeToWaitInSeconds < settings["predictionLatency"]:
        logger.info("Too close but if not sent this period then give it a shot...")
        if settings["lastSent"] < timestamp - (
            settings["referencePeriod"] / settings["framerate"]
        ):
            logger.success("Trigger sent!")
            sendIt = 1
        else:  # if already sent this period, start prediction for next cycle
            logger.info("Trigger not sent! Starting prediction for next cycle.")
            timeToWaitInSeconds += settings["referencePeriod"] / settings["framerate"]
    elif (timeToWaitInSeconds - (framerateFactor / settings["framerate"])) < settings[
        "predictionLatency"
    ]:
        logger.success("We won't be able to do another calculation... so trigger sent!")
        sendIt = 2
    else:
        pass

    if sendIt > 0 and settings["lastSent"] > (
        timestamp - ((settings["referencePeriod"] / settings["framerate"]) / 2)
    ):
        # if we've done any triggering in the last half a cycle, don't trigger
        # this is quite different from JTs approach
        # where he follows which cycle we last triggered on
        logger.info(
            "Trigger type {0} at {1}\tDROPPED", sendIt, timestamp + timeToWaitInSeconds
        )
        sendIt = 0
    elif sendIt > 0:
        logger.success("Trigger to be sent, updating `settings['lastSent']`.")
        settings["lastSent"] = timestamp

    return timeToWaitInSeconds, sendIt, settings
