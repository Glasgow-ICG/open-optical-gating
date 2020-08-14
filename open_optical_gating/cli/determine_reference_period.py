"""Functions to establish a reference heartbeat/period
as used from prospective optical gating."""

# Module imports
import os
import numpy as np
from skimage import io
import j_py_sad_correlation as jps
from loguru import logger
from datetime import datetime

# Local
from . import parameters
from . import prospective_optical_gating as pog


def establish(sequence, settings, padding=True):
    # TODO: JT writes: here and elsewhere, why does this return settings back again?
    # I’m 99% sure the original settings object will be modified, so returning a new object seems confusing to me.
    # I can see that returning it could be a reminder that it is changed by the function, but equally it implies to me that
    # if the caller saved the return value into a *different* variable, the *original* object would be unmodified (which is not the case).
    # (We can't resolve this by doing a deep copy(), because the object is a large one that contains frame data.
    #  I would be wary of a shallow copy, because that's just storing up confusion for the future).
    # I will have a think and try and come up with a solution I like for this general issue.
    # -> UPDATE: actually, this is bonkers. As far as I can see, all that is updated is the reference period
    """ Attempt to establish a reference period from a sequence of recently-received frames.
        Parameters:
            sequence    list of ndarrays Sequence of recently-received frame pixel arrays (in chronological order)
            settings    dict             Parameters controlling the sync algorithms
            padding     bool             Should we pad the returned sequence with extra frames
        Returns:
            List of frame pixel arrays that form the reference sequence (or None).
    """
    referenceFrameIdx, settings = establish_indices(sequence, settings, padding)
    logger.trace("Idx: {0}", sequence[referenceFrameIdx].shape)
    return sequence[referenceFrameIdx], settings


def establish_indices(sequence, settings, padding=True):
    """ Establish the list indices representing a reference period, from a given input sequence.
        Parameters:
            sequence    list of ndarrays Sequence of recently-received frame pixel arrays (in chronological order)
            settings    dict             Parameters controlling the sync algorithms
            padding     bool             Should we pad the returned sequence with extra frames
        Returns:
            List of indices that form the reference sequence (or None).
    """

    periods = []
    for i in range(1, len(sequence)):
        frame = sequence[i, :, :]
        pastFrames = sequence[: (i - 1), :, :]
        logger.trace("Running for frame {0}", i)

        # Calculate Diffs between this frame and previous frames in the sequence
        diffs = jps.sad_with_references(frame, pastFrames)

        # Calculate Period based on these Diffs
        period = calculate_period_length(diffs)
        if period != -1:
            periods.append(period)

        # If we have a valid period, extract the frame indices associated with this period, and return them
        # The conditions here are empirical ones to protect against glitches where the heuristic
        # period-determination algorithm finds an anomalously short period.
        # JT TODO: The four conditions seem to be pretty similar/redundant. I wrote these many years ago,
        #  and have just left them as they "ain't broke". They should really be tidied up though.
        #  One thing I can say is that the reason for the *two* tests for >6 have to do with the fact that
        #  we are establishing the period based on looking back from the *most recent* frame, but then actually
        #  and up taking a period from a few frames earlier, since we also need to incorporate some extra padding frames.
        #  That logic could definitely be improved and tidied up - we should probably just
        #  look for a period starting numExtraRefFrames from the end of the sequence...
        # TODO: JT writes: logically these tests should probably be in calculate_period_length, rather than here
        if (period != -1
            and len(periods) >= (5 + (2 * settings["numExtraRefFrames"]))
            and period > 6
            and (len(periods) - 1 - settings["numExtraRefFrames"]) > 0
            and (periods[len(periods) - 1 - settings["numExtraRefFrames"]]) > 6
        ):
            periodToUse = periods[len(periods) - 1 - settings["numExtraRefFrames"]]
            logger.success("Found a period I'm happy with: {0}".format(periodToUse))

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
    """ Attempt to determine the period of one heartbeat, from the diffs array provided. The period will be measured backwards from the most recent frame in the array
        Parameters:
            diffs    ndarray    Diffs between latest frame and previously-received frames
        Returns:
            Period, or -1 if no period found
        """
    
    # TODO: JT writes: the comment below can probably be removed now I have written the above function header. BUT, can somebody confirm or deny
    # the bit about "the list is in reverse order"? That does not seem consistent with e.g. "bestMatchEntry = diffs.size - bestMatchPeriod", "score = diffs[diffs.size - d]" etc.
    # Note that I have added comments about "chronological order" to earlier functions, above, based on my code-reading (which matches my expectations),
    # but I started to doubt myself here when I found an explicit comment saying "reverse order"...
    
    # Calculate the heart period (with sub-frame interpolation) based on a provided list of comparisons between the current frame and previous frames. The list is in reverse order (i.e. difference with most recent frame comes first)
    bestMatchPeriod = estimate_integer_period_length(diffs)
    bestMatchEntry = diffs.size - bestMatchPeriod

    if bestMatchPeriod == -1:
        return -1

    # TODO: JT writes: isn't it slightly weird that the v_fitting function lives in the POG module?
    # Feel free just to delete this comment if you don't have any better ideas, though!
    interpolatedMatchEntry = (
        bestMatchEntry
        + pog.v_fitting(
            diffs[bestMatchEntry - 1], diffs[bestMatchEntry], diffs[bestMatchEntry + 1]
        )[0]
    )

    return diffs.size - interpolatedMatchEntry


def estimate_integer_period_length(diffs):
    # Unlike JTs codes, this function currently only supports determining the period for a *one* beat sequence.
    # It therefore also only supports determining a period which ends with the final frame in the diffs sequence.
    # TODO: JT writes: this function needs a proper function header comment, but first we need to think about whether there's any point having
    # this as a separate function. Is there any purpose to separating this out from calculate_period_length?
    if diffs.size < 2:
        logger.debug("Not enough diffs, returning -1")
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
    # TODO: JT writes: I assume the above comment refers to the list of values. Ugh. I agree, this really needs tidying into a dictionary
    # or, dare I say it, a class(!). This might be where your and my class philosophies do diverge a little; it's a borderline one for me even.
    # But, I think it's a reasonable concept to have a class instance (lifetime could be while we are in the establish-period state...)
    # whose job it is to determine the period. There is internal state to be maintained (as demonstrated by the passing-around of 'values'),
    # and when running live (not just determining period in one go, from one big historical frame buffer) we might potentially want
    # some of that state to persist from one received frame to the next. So there are benefits to maintaining, and encapsulating, that state in one object.
    for d in range(2, diffs.size):
        logger.trace(d)
        score = diffs[diffs.size - d]
        got, values = gotScoreForDelta(score, d, values)
        if got:
            return values[5]

    logger.debug("I didn't find a whole period, returning -1")
    return -1  # catch if doesn't find a period


def gotScoreForDelta(score, d, values):
    # TODO: JT writes: this function needs a proper function header comment, but let's wait until the "WTF"/"Ugh" comments above has been addressed, because that will require a refactor I think.
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


def save_period(reference_period, period_dir="~/"):
    # TODO: JT writes: Needs an explanation of parameters and purpose, and what the isinstance test is about
    # (which I find a bit weird - are we expecting that reference_period might be an int, or is that just misuse of the function?)
    '''Function to save a reference period in a time-stamped folder.'''
    dt = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    os.makedirs(os.path.join(period_dir, dt), exist_ok=True)

    # Saves the period
    if isinstance(reference_period, int) == False:
        for i, frame in enumerate(reference_period):
            io.imsave(os.path.join(period_dir, dt, "{0:03d}.tiff".format(i)),frame)
