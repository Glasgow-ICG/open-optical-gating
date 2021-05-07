"""Settings functions for open optical gating system."""

# TODO: JT writes: Needs a more extensive comment on what “parameters” actually are. They are a mix of config parameters and current status (e.g. reference frames)...
# What is the relationship between these and the other "settings" dictionary that gets passed around?
# Might make sense to rename this "pog_settings.py" or similar, to be consistent with the variable name that is used for them, and to help avoid confusion
# ... but I expect to tackle all of this in a refactor, so probably just leave this comment as a reminder for now.

# TODO: JT writes: should also consider whether this makes sense as a separate object rather than just properties of optical_gater_server.
# ... or conversely perhaps prospective_optical_gating should be methods on this POGSettings object...!?
# Should I *remove* any variables from this such as lastSent, referenceFrame, reference_period, barrierFrameLookup(?) etc? I need to think about all this...

# TODO: JT writes: numExtraRefFrames should really be a global constant, not a parameter in settings.
# Really the only reason that parameter exists at all in the C code is to self-document all the +-2 arithmetic that would otherwise appear.
# In the C code it is declared as a const int.

## Imports
import numpy as np


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
                    "numExtraRefFrames": 2,
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
        self.special_list = [ "referenceFrameCount", "targetSyncPhase", "oga_reference_value" ]
    
    def __getitem__(self, key):
        if key in self.special_list:
            if key == "referenceFrameCount":
                # Number of reference frames including padding
                # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
                return int(self["reference_period"] + 1) \
                       + (2 * self["numExtraRefFrames"])
            elif key == "targetSyncPhase":
                # Target phase in rads
                if self["reference_period"] > 0.0:
                    return 2 * np.pi * (self["referenceFrame"] / self["reference_period"])
                else:
                    return 0
            elif key == "oga_reference_value":
                # Target frame index as understood by optical_gating_alignment
                if self["reference_period"] > 0.0:
                    return self["oga_resampled_period"] * (self["referenceFrame"] / self["reference_period"])
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
                   (val - self["numExtraRefFrames"])
                   % self["reference_period"]
                   ) + self["numExtraRefFrames"]

        return val
    
    def __setitem__(self, key, val):
        if key == "oga_reference_value":
            # This is a derived key, but the caller is permitted to set it.
            # Internally we just translate to referenceFrame, which is our definitive value
            assert(self["reference_period"] > 0.0)
            translated = self["reference_period"] * (val / self["oga_resampled_period"])
            translated = translated % self["reference_period"]
            dict.__setitem__(self, "referenceFrame", translated)
        elif key in self.special_list:
            raise KeyError("Attempting to set read-only parameter '{0}'".format(key))
        else:
            dict.__setitem__(self, key, val)
