"""Settings functions for open optical gating system."""

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
    referencePeriod=0.0,
    barrierFrame=0.0,
    extrapolationFactor=1.5,
    maxReceivedFramesToStore=260,
    maxFramesForFit=32,
    minFramesForFit=3,
    predictionLatency=15,
    referenceFrame=0.0,
    numExtraRefFrames=2,
):
    """Function to intialise our custom settings dict with sensible pre-sets."""
    parameters = {}
    parameters.update({"drift": drift})  # starting drift corrections
    parameters.update({"framerate": framerate})  # starting est frame rate
    parameters.update(
        {"referencePeriod": referencePeriod}
    )  # reference period in frames
    if barrierFrame > 0.0:
        parameters.update(
            {
                "barrierFrame": ((barrierFrame - numExtraRefFrames) % referencePeriod)
                + numExtraRefFrames
            }
        )  # barrier frame in frames
    else:
        parameters.update({"barrierFrame": barrierFrame})
    parameters.update({"extrapolationFactor": extrapolationFactor})
    parameters.update(
        {"maxReceivedFramesToStore": maxReceivedFramesToStore}
    )  # maximum number of frames to stores, used to prevent memory filling
    parameters.update(
        {"maxFramesForFit": maxFramesForFit}
    )  # frames to fit for prediction (max)
    parameters.update(
        {"minFramesForFit": minFramesForFit}
    )  # frames to fit for prediction (min)
    parameters.update(
        {"predictionLatency": predictionLatency}
    )  # prediction latency in milliseconds
    if referenceFrame > 0.0:
        parameters.update(
            {"referenceFrame": referenceFrame % referencePeriod}
        )  # target phase in frames
    else:
        parameters.update({"referenceFrame": referenceFrame})
    parameters.update({"numExtraRefFrames": numExtraRefFrames})  # padding number

    # automatically added keys
    # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
    parameters.update(
        {
            "referenceFrameCount": int(parameters["referencePeriod"] + 1)
            + (2 * parameters["numExtraRefFrames"])
        }
    )  # number of reference frames including padding
    if referencePeriod > 0.0:
        parameters.update(
            {
                "targetSyncPhase": 2
                * np.pi
                * (parameters["referenceFrame"] / parameters["referencePeriod"])
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
    referencePeriod=None,
    barrierFrame=None,
    extrapolationFactor=None,
    maxReceivedFramesToStore=None,
    maxFramesForFit=None,
    minFramesForFit=None,
    predictionLatency=None,
    referenceFrame=None,
    numExtraRefFrames=None,
):
    """Function to update our custom settings dict with sensible pre-sets.
    Note: users should not use parameters.update(), i.e. a dictionary update
    as it will not automatically update certain keys.
    """
    if drift is not None:
        parameters["drift"] = drift
    if framerate is not None:
        parameters["framerate"] = framerate
    if referencePeriod is not None:
        parameters["referencePeriod"] = referencePeriod
    if numExtraRefFrames is not None:
        parameters["numExtraRefFrames"] = numExtraRefFrames
    if barrierFrame is not None:
        parameters["barrierFrame"] = (
            (barrierFrame - parameters["numExtraRefFrames"])
            % parameters["referencePeriod"]
        ) + parameters["numExtraRefFrames"]
    if extrapolationFactor is not None:
        parameters["extrapolationFactor"] = extrapolationFactor
    if maxReceivedFramesToStore is not None:
        parameters["maxReceivedFramesToStore"] = maxReceivedFramesToStore
    if maxFramesForFit is not None:
        parameters["maxFramesForFit"] = maxFramesForFit
    if minFramesForFit is not None:
        parameters["minFramesForFit"] = minFramesForFit
    if predictionLatency is not None:
        parameters["predictionLatency"] = predictionLatency
    if referenceFrame is not None:
        parameters["referenceFrame"] = referenceFrame % parameters["referencePeriod"]

    # automatically added keys
    # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
    parameters.update(
        {
            "referenceFrameCount": int(parameters["referencePeriod"] + 1)
            + (2 * parameters["numExtraRefFrames"])
        }
    )  # number of reference frames including padding
    # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
    parameters.update(
        {
            "targetSyncPhase": 2
            * np.pi
            * (parameters["referenceFrame"] / parameters["referencePeriod"])
        }
    )  # target phase in rads

    return parameters
