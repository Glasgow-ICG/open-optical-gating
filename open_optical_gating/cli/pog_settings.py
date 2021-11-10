""" An object encapsulating state for the optical gating algorithms
    
    Note that at the moment this contains both user-accessible parameters that affect algorithm behaviour,
    and actual *state* (e.g. the current reference frames).
    There is also some conceptual overlap between this object and e.g. the prospective_optical_gating.py module.
    Note that some pog.xxx functions need to be passed a settings object, while others are standalone.
    The structure of all this would probably benefit from some further refactoring and encapsulation...
"""

# TODO: framerate is currently used in POG logic, but duplicates the "brightfield_framerate" information in the gating server.
# We should think about whether we want the duplication, and indeed whether it needs to be kept updated when it changes,
# and/or whether we would be better off just deducing the framerate from the timestamps of recently-received frames

## Python imports
import numpy as np

# Local imports
#
# This is only required for a couple of accesses to numExtraRefFrames.
# It's possible that that is a hint that things should be refactored slightly,
# so that this current module doesn't need to know about that.
# It's only really used as a safety check, anyway.
from . import prospective_optical_gating as pog

def load():
    """Function to load settings from a json file into our custom dict."""
    return


def save():
    """Function to save persistent settings from our custom dict as a json file."""
    return

class POGSettings(dict):
    def __init__(self, params=None):
        defaults = {
                    "drift": [0, 0],
                    "framerate": 80,
                    "ref_frames": None,
                    "reference_period": 0.0,
                    # barrier frame in frames
                    "barrierFrame": 0.0,
                    "extrapolationFactor": 1.5,
                    # Used to prevent over-use of memory (and CPU) in cases where we fail to lock on,
                    # where we would just keep processing a longer and longer frame history buffer.
                    "maxReceivedFramesToStore": 1000,
                    # frames to fit for prediction (max)
                    "maxFramesForFit": 32,
                    # frames to fit for prediction (min)
                    "minFramesForFit": 3,
                    "prediction_latency_s": 0.015,
                    "referenceFrame": 0.0,
                    "phase_stamp_only": False,
                    "minPeriod": 5,
                    "lowerThresholdFactor": 0.5,
                    "upperThresholdFactor": 0.75,
                    # Time (s) when trigger was last sent
                    "lastSent": 0,
                    # Resampling factor to be used in optical_gating_alignment
                    "oga_resampled_period": 80,
                   }
        super(POGSettings, self).__init__(defaults)
        # Replace default values with anything specified in the settings dictionary passed to our constructor
        if params is not None:
            dict.update(params)

        # These are read-only parameters that we handle as derived quantities
        self.special_list = [ "referenceFrameCount", "targetSyncPhase" ]
        self.special_list2 = [ "referenceFrame", "barrierFrame" ]

    def __getitem__(self, key):
        if key in self.special_list:
            if key == "referenceFrameCount":
                # Number of reference frames including padding
                return self["ref_frames"].shape[0]
            elif key == "targetSyncPhase":
                # Target phase in rads
                if self["reference_period"] > 0.0:
                    return 2 * np.pi * (self["referenceFrame"] / self["reference_period"])
                else:
                    return 0
            else:
                raise NotImplementedError("Read-only key {0} not implemented".format(key))
        
        # Standard case: access the value from the dictionary
        val = dict.__getitem__(self, key)
        
        if (key == "referenceFrame") and (val > 0.0):
            # Ensure reference frame does not lie outside the valid range of references,
            # because that might cause big problems for the gating code.
            # Note that we do this in getitem, not setitem, in case the reference frames
            # change after referenceFrame is set
            val = val % self["reference_period"]
        if (key == "barrierFrame") and (val > 0.0):
            # Impose a similar range limitation on barrierFrame
            val = (
                   (val - pog.numExtraRefFrames)
                   % self["reference_period"]
                   ) + pog.numExtraRefFrames

        return val
    
    def __setitem__(self, key, val):
        if key in self.special_list:
            raise KeyError("Attempting to set read-only parameter '{0}'".format(key))
        elif key in self.special_list2:
            raise KeyError("Parameter '{0}' must be set through a dedicated setter function".format(key))
        else:
            dict.__setitem__(self, key, val)

    def set_reference_frames(self, ref_frames, period_to_use):
        # Note that we may be passed ref_frames as a list, and this is helpful because it means we could still access
        # the PixelArray metadata at this point if we wished.
        # However, long-term we want to store a 3D array because that is what OGA expects to work with.
        # We therefore make that conversion here
        # Note that this function does not do anything to self["referenceFrame"].
        # It is up to the caller to set or update that as they wish
        ref_frames = np.array(ref_frames)
        self["ref_frames"] = ref_frames
        self["reference_period"] = period_to_use

    def set_reference_and_barrier_frame(ref, barrier):
        self["referenceFrame"] = ref
        self["barrierFrame"] = barrier
        # Precalculate a lookup of the barrier frames
        self["framesForFitLookup"] = pog.determine_barrier_frame_lookup(self)
