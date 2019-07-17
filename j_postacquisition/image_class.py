import numpy as np
import warnings
import operator
from phase_wrap_interpolation import *

# Simple class which serves as a container for image data and various bits of metadata
# (which are mostly derived from the associated plist file)
#
# === NOTE ===
#
# In general you should not directly access any variables starting with an underscore
# (e.g. _frameMetadata, _plistPath etc) from your own code. It is sometimes necessary
# to do so (especially if re-saving files and metadata), but try and use existing
# functions for that where possible.
#
# For accessing metadata properties, the intention is that wherever possible
# you should use the function accessors such as binning_code().
# If there is not an accessor for a property you want to read, then add one!
# The reason for doing it this way is so that, if in future the metadata format changes again,
# then your code will continue to run and I will just change the code in this one place here.
# That said, in some cases (especially when editing plist entries) I have had to write code
# elsewhere that explicitly accesses the _frameMetadata dictionary (for example).
# Code that does that may break and need fixing if the plist format changes in future.
#
class ImageClass(object):
    def __init__(self):
        return
    def fi(self):
        return self.frameIndex
    def ph(self):
        return self.phase

class ImageClass_v3(ImageClass):
    # Frame properties
    def frameKeyPath(self, key):
        val = self._frameMetadata
        for k in key.split('.'):
            val = val[k]
        return val
    def postacquisition_phase(self):
        # Default to -1 if key is absent
        if ('postacquisition_phase' in self._frameMetadata):
            return self.frameKeyPath('postacquisition_phase')
        else:
            return -1.0
    def phase_from_offline_sync_analysis(self):
        # Default to -1 if key is absent
        if ('phase_from_offline_sync_analysis' in self._frameMetadata):
            return self.frameKeyPath('phase_from_offline_sync_analysis')
        else:
            return -1.0
    def z_scan(self):
        # In future I might move this to global metadata.
        # If I do, then I also need to update image_loading.py which has a line:
        #        if ('z_scan' in image._frameMetadata)
        return self.frameKeyPath('z_scan')
    def stage_positions(self):
        return self.frameKeyPath('stage_positions')
    def time_exposed(self):
        return self.frameKeyPath('time_exposed')
    def bestTimestamp(self):
        # returns the most appropriate timestamp
        # see JTs universalComputerTimestamp in C
        timestamp = None
        try:
            timestamp = self.frameKeyPath('time_exposed')
            return timestamp
        except Exception:
            warnings.warn('Key \'time_exposed\' not found. Resorting to alternatives.')
            try:
                timestamp = self.frameKeyPath('time_received3')
                return timestamp
            except Exception:
                warnings.warn('Key \'time_received3\' not found. Resorting to alternatives.')
                try:
                    timestamp = self.frameKeyPath('time_received')
                    return timestamp
                except Exception:
                    raise LookupError('Key \'time_received\' not found. No time stamp found.')


    # Camera properties
    def binning_code(self):
        return self._globalMetadata['binning_code']

class ImageClass_v1or2(ImageClass_v3):
    # For metadata version 2 (and version 1) the camera metadata is mixed in with the frame-specific metadata
    def binning_code(self):
        return self.frameKeyPath('binning_code')


def InvalidPhase(fluorPhases):
    invalid = fluorPhases > 6.9

def InterpolateForImagePhases(fluorImages, knownTimes, knownPhases):
    # Interpolate to get fluor phases, using the known information from the brightfield data
    fluorTimes = []
    for i in range(len(fluorImages)):
        fluorTimes.append(fluorImages[i].timestamp)
    fluorTimes = np.array(fluorTimes)
    fluorPhases = interpolate_with_phase_wrap(fluorTimes, knownTimes, knownPhases)

    # Try setting all nans to phase 7,  that may help with sorting etc
    for i in range(len(fluorPhases)):
        if (np.isnan(fluorPhases[i])):
            fluorPhases[i] = 7
    
    for i in range(len(fluorImages)):
        fluorImages[i].phase = fluorPhases[i]
    
    sortedFluorImages = sorted(fluorImages, key=operator.attrgetter('phase'))
    print(len(np.where(InvalidPhase(fluorPhases))[0]), '/', len(fluorPhases), 'failed to recover a phase')

    return sortedFluorImages
