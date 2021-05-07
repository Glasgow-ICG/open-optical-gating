"""Settings functions for open optical gating system."""

# TODO: JT writes: Needs a more extensive comment on what “parameters” actually are. They are a mix of config parameters and current status (e.g. reference frames)...
# What is the relationship between these and the other "settings" dictionary that gets passed around?
# Might make sense to rename this "pog_settings.py" or similar, to be consistent with the variable name that is used for them, and to help avoid confusion
# ... but I expect to tackle all of this in a refactor, so probably just leave this comment as a reminder for now.

## Imports
import numpy as np


def load():
    """Function to load settings from a json file into our custom dict."""
    return


def save():
    """Function to save persistent settings from our custom dict as a json file."""
    return


def initialise(
    drift=[0, 0],
    framerate=80,
    reference_period=0.0,
    barrierFrame=0.0,
    extrapolationFactor=1.5,
    maxReceivedFramesToStore=1000,
    maxFramesForFit=32,
    minFramesForFit=3,
    prediction_latency_s=0.015,
    referenceFrame=0.0,
    numExtraRefFrames=2,
    phase_stamp_only=False,
    minPeriod=5,
    lowerThresholdFactor=0.5,
    upperThresholdFactor=0.75,
):
    """Function to initialise our custom settings dict with sensible pre-sets."""
    parameters = {}
    parameters.update({"drift": drift})  # starting drift corrections
    parameters.update({"framerate": framerate})  # starting est frame rate
    parameters.update(
        {"reference_period": reference_period}
    )  # reference period in frames
    if barrierFrame > 0.0:
        parameters.update(
            {
                "barrierFrame": ((barrierFrame - numExtraRefFrames) % reference_period)
                + numExtraRefFrames
            }
        )  # barrier frame in frames
    else:
        parameters.update({"barrierFrame": barrierFrame})
    parameters.update({"extrapolationFactor": extrapolationFactor})
    parameters.update(
        {"maxReceivedFramesToStore": maxReceivedFramesToStore}
    )  # maximum number of frames to store while establishing the period.
    # Used to prevent over-use of memory (and CPU) in cases where we fail to lock on,
    # where we would just keep processing a longer and longer frame history buffer.
    parameters.update(
        {"maxFramesForFit": maxFramesForFit}
    )  # frames to fit for prediction (max)
    parameters.update(
        {"minFramesForFit": minFramesForFit}
    )  # frames to fit for prediction (min)
    parameters.update(
        {"prediction_latency_s": prediction_latency_s}
    )  # prediction latency in seconds
    if referenceFrame > 0.0:
        parameters.update(
            {"referenceFrame": referenceFrame % reference_period}
        )  # target phase in frames
    else:
        parameters.update({"referenceFrame": referenceFrame})
    parameters.update({"numExtraRefFrames": numExtraRefFrames})  # padding number
    parameters.update({"phase_stamp_only": phase_stamp_only})
    parameters.update({"minPeriod": minPeriod})
    parameters.update({"lowerThresholdFactor": lowerThresholdFactor})
    parameters.update({"upperThresholdFactor": upperThresholdFactor})

    # automatically added keys
    # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
    parameters.update(
        {
            "referenceFrameCount": int(parameters["reference_period"] + 1)
            + (2 * parameters["numExtraRefFrames"])
        }
    )  # number of reference frames including padding
    if reference_period > 0.0:
        parameters.update(
            {
                "targetSyncPhase": 2
                * np.pi
                * (parameters["referenceFrame"] / parameters["reference_period"])
            }
        )  # target phase in rads
    else:
        parameters.update({"targetSyncPhase": 0})  # target phase in rads
    parameters.update({"lastSent": 0.0})
    # parameters.update({'frameToUseArray':[0]})#this should be created locally when needed

    return parameters


def update(
    parameters,
    drift=None,
    framerate=None,
    reference_period=None,
    barrierFrame=None,
    extrapolationFactor=None,
    maxReceivedFramesToStore=None,
    maxFramesForFit=None,
    minFramesForFit=None,
    prediction_latency_s=None,
    referenceFrame=None,
    numExtraRefFrames=None,
    phase_stamp_only=None,
    minPeriod=None,
    lowerThresholdFactor=None,
    upperThresholdFactor=None,
):
    """Function to update our custom settings dict with sensible pre-sets.
    Note: users should not use parameters.update(), i.e. a dictionary update
    as it will not automatically update certain keys.
    """
    # TODO: JT writes: what is the *role* of this function? The “note” in the comment is not clear to me.
    # Under what circumstances should this function used, by whom, and why? What is the alternative?
    # JT reminder for self: Chas explains that it's because of the auto-updated variables targetSyncPhase and referenceFrameCount
    # This should indeed all get cleaned up, once we have a proper refactor into a Sync Analyzer class.
    if drift is not None:
        parameters["drift"] = drift
    if framerate is not None:
        parameters["framerate"] = framerate
    if reference_period is not None:
        parameters["reference_period"] = reference_period
    if numExtraRefFrames is not None:
        parameters["numExtraRefFrames"] = numExtraRefFrames
    if phase_stamp_only is not None:
        parameters["phase_stamp_only"] = phase_stamp_only
    if minPeriod is not None:
        parameters["minPeriod"] = minPeriod
    if lowerThresholdFactor is not None:
        parameters["lowerThresholdFactor"] = lowerThresholdFactor
    if upperThresholdFactor is not None:
        parameters["upperThresholdFactor"] = upperThresholdFactor

    if barrierFrame is not None:
        parameters["barrierFrame"] = (
            (barrierFrame - parameters["numExtraRefFrames"])
            % parameters["reference_period"]
        ) + parameters["numExtraRefFrames"]
    if extrapolationFactor is not None:
        parameters["extrapolationFactor"] = extrapolationFactor
    if maxReceivedFramesToStore is not None:
        parameters["maxReceivedFramesToStore"] = maxReceivedFramesToStore
    if maxFramesForFit is not None:
        parameters["maxFramesForFit"] = maxFramesForFit
    if minFramesForFit is not None:
        parameters["minFramesForFit"] = minFramesForFit
    if prediction_latency_s is not None:
        parameters["prediction_latency_s"] = prediction_latency_s
    if referenceFrame is not None:
        parameters["referenceFrame"] = referenceFrame % parameters["reference_period"]

    # automatically added keys
    # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
    parameters.update(
        {
            "referenceFrameCount": int(parameters["reference_period"] + 1)
            + (2 * parameters["numExtraRefFrames"])
        }
    )  # number of reference frames including padding
    # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
    parameters.update(
        {
            "targetSyncPhase": 2
            * np.pi
            * (parameters["referenceFrame"] / parameters["reference_period"])
        }
    )  # target phase in rads

    return parameters
