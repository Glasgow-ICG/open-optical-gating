import numpy as np

def interpolate_with_phase_wrap(unknownTimes, knownTimes, knownPhases, phaseWrapValue = 2*np.pi):
    # This function takes a SORTED array of knownTimes, with associated knownPhases,
    # and does linear interpolation to return the phases associated with the array of unknownTimes.
    # It understands the concept of a phase wrap (which by default is at 2*pi but could be specified
    # to be at some other value e.g. 1.0)
    
    # Work out where 'unknownTimes' fit in to the sorted list of 'knownTimes'
    afterIndices = np.searchsorted(knownTimes, unknownTimes)
    
    # Identify locations where unknownTimes lie outside the range of knownTimes
    # (where we cannot interpolate, although we could in principle extrapolate)
    problemLocations = np.where((afterIndices == 0) | (afterIndices == knownTimes.size))
    # For now, just set those locations to lie between index 0 and index 1 in knownTimes.
    # This is totally wrong, especially for values at the END of knownTimes!
    # However, all we are doing for now is ensuring Python does not report an error.
    # We will be setting these values to something else anyway, later on
    afterIndices[problemLocations] = 1
    
    # Identify values to use in our interpolation
    beforeIndices = afterIndices - 1
    afterTimes = knownTimes[afterIndices]
    beforeTimes = knownTimes[beforeIndices]
    afterPhases = knownPhases[afterIndices]
    beforePhases = knownPhases[beforeIndices]
    
    # Initially do a naive linear interpolation between knownPhases
    # This will not behave as desired in the presence of a phase wrap
    frac = (unknownTimes - beforeTimes) / (afterTimes - beforeTimes)
    unknownPhases = frac * afterPhases + (1 - frac) * beforePhases
    
    # Now do special handling of those cases where there is a phase wrap
    wrapIndices = np.where((beforePhases > 0.75 * phaseWrapValue) & (afterPhases < 0.25 * phaseWrapValue))
    unknownPhases[wrapIndices] = (frac[wrapIndices] * (afterPhases[wrapIndices] + phaseWrapValue) + (1 - frac[wrapIndices]) * beforePhases[wrapIndices]) % phaseWrapValue
    
    # Handle cases where unknownTimes are outside the range of knownTimes
    # (see detailed comments above)
    # Set them to NaN - it's down to the caller to deal with those as they come up.
    unknownPhases[problemLocations] = np.nan
    
    return unknownPhases