"""Functions to establish a reference heartbeat/period
as used from prospective optical gating."""

# Module imports
import numpy as np
import j_py_sad_correlation as jps
import loguru as logger

# Local
import parameters


def establish(sequence, settings, padding=True):
    """Establish a reference period from a sequence.
    Do the processing required for the initial establishing of the heart period
    (and associated selection of one period's worth of reference frames).
    Establish the period for the cycle leading up to this current frame.
    """
    # Assumes sequence is a list of numpy arrays
    # This returns the frames (c.f. establish_indices)
    referenceFrameIdx, settings = establish_indices(sequence, settings, padding)
    logger.trace("Idx: {0}", sequence[referenceFrameIdx].shape)
    return sequence[referenceFrameIdx], settings


def establish_indices(sequence, settings, padding=True):
    """Establish the indices of a reference period from a sequence.
    Do the processing required for the initial establishing of the heart period
    (and associated selection of one period's worth of reference frames).
    Establish the period for the cycle leading up to this current frame.
    """
    # Assumes sequence is a list of numpy arrays

    periods = []
    for i in range(1, len(sequence)):
        frame = sequence[i, :, :]
        pastFrames = sequence[: (i - 1), :, :]
        logger.trace("Running for frame {0}", i)

        # Calculate Diffs
        diffs = jps.sad_with_references(frame, pastFrames)

        # Calculate Period for Diffs
        period = calculate_period_length(diffs)
        if period != -1:
            periods.append(period)

        # Get One Periods Frames
        if (
            period != -1
            and len(periods) >= (5 + (2 * settings["numExtraRefFrames"]))
            and period > 6
            and (len(periods) - 1 - settings["numExtraRefFrames"]) > 0
            and (periods[len(periods) - 1 - settings["numExtraRefFrames"]]) > 6
        ):
            logger.success("Found a period I'm happy with")
            periodToUse = periods[len(periods) - 1 - settings["numExtraRefFrames"]]

            settings = parameters.update(
                settings, referencePeriod=periodToUse
            )  # automatically does referenceFrameCount an targetSyncPhase
            if padding:
                # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
                numRefs = int(periodToUse + 1) + (2 * settings["numExtraRefFrames"])
                return np.arange(len(pastFrames) - numRefs, len(pastFrames)), settings
            else:
                # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
                numRefs = int(periodToUse + 1) + settings["numExtraRefFrames"]
                return (
                    np.arange(
                        len(pastFrames) - numRefs,
                        len(pastFrames) - settings["numExtraRefFrames"],
                    ),
                    settings,
                )

    logger.critical("I didn't find a period I'm happy with!")
    return None, settings


def calculate_period_length(diffs):
    # Calculate the heart period (with sub-frame interpolation) based on a provided list of comparisons between the current frame and previous frames. The list is in reverse order (i.e. difference with most recent frame comes first)
    bestMatchPeriod = estimate_period_length(diffs)
    bestMatchEntry = diffs.size - bestMatchPeriod

    if bestMatchPeriod == -1:
        return -1

    interpolatedMatchEntry = (
        bestMatchEntry
        + pog.v_fitting(
            diffs[bestMatchEntry - 1], diffs[bestMatchEntry], diffs[bestMatchEntry + 1]
        )[0]
    )

    return diffs.size - interpolatedMatchEntry


def estimate_period_length(diffs):
    # unlike JTs codes I assume 1 period with a start frame of 1
    if diffs.size < 2:
        logger.warning("Not enough diffs, returning -1")
        return -1

    score = diffs[diffs.size - 1]
    values = [
        score,
        score,
        score,
        score,
        score,
        0,
        1,
        1,
    ]  # list of values needed in the gotScoreForDelta function: minScore, maxScore, totalScore, meanScore, minSinceMax, deltaForMinSinceMax, stage, numScores
    # TODO WTF, can this be simplified?
    for d in range(2, diffs.size):
        logger.trace(d)
        score = diffs[diffs.size - d]
        got, values = gotScoreForDelta(score, d, values)
        if got:
            return values[5]

    logger.warning("I didn't find a whole period, returning -1")
    return -1  # catch if doesn't find a period


def gotScoreForDelta(score, d, values):
    # values = (minScore, maxScore, totalScore, meanScore, minSinceMax, deltaForMinSinceMax, stage, numScores)

    values[2] += score  # totalScore
    values[7] += 1  # numScores

    lowerThresholdScore = values[0] + (values[1] - values[0]) / 2  # minScore,maxScore
    upperThresholdScore = (
        values[0] + (values[1] - values[0]) * 3 / 4
    )  # minScore,maxScore
    logger.debug(
        "Lower Threshold:\t{0:.4f};\tUpper Threshold:\t{1:.4f}",
        lowerThresholdScore,
        upperThresholdScore,
    )

    if score < lowerThresholdScore and values[6] == 1:  # stage
        logger.info("Stage 1: Under lower threshold; Moving to stage 2")
        values[6] = 2  # stage

    if score > upperThresholdScore and values[6] == 2:  # stage
        # TODO: speak to JT about the 'final condition'
        logger.info(
            "Stage 2: Above upper threshold; Returning period of {0}", values[5]
        )
        values[6] = 3  # stage
        return True, values

    if score > values[1]:  # maxScore
        logger.info("New max score: {0} > {1}. Resetting to stage 1.", score, values[1])
        values[1] = score  # maxScore
        values[4] = score  # minSinceMax
        values[5] = d  # deltaForMinSinceMax
        values[6] = 1  # stage
    elif score != 0 and (values[0] == 0 or score < values[0]):  # minScore
        logger.debug("New minimum score of {0}", score)
        values[0] = score  # minScore

    if score < values[4]:  # minSinceMax
        logger.debug("New minimum score ({0}) since maximum of {1}", score, values[1])
        values[4] = score  # minSinceMax
        values[5] = d  # deltaForMinSinceMax

    # Note this is only updated AFTER we have done the other processing (i.e. the mean score used does NOT include the current delta)
    values[3] = values[2] / values[7]
    # meanScore,totalScore,numeScores

    return False, values
