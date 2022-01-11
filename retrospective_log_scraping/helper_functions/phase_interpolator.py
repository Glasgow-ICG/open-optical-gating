import numpy as np

def TriggerPhaseInterpolation(timeStamps, states, phases, triggerTimes, targetPhases):
    """
    A function to interpolate phase between brightfield frames in order to estimate the phase
    at which fluorescent images were captured. 
    
    Inputs:
        timeStamps = array of BF frame timestamps
        phases = array of BF frame phases
        triggerTimes = array of sent trigger times
        targetPhases = array of the target phase at each BF timestamp
    
    Outputs:
        unwrappedPhases = array of interpolated unwrapped phase at each sent trigger time
        phaseErrors = array of phase error (interpolated phase - target phase) at each trigger time
    """
    
    unwrappedPhases = []
    phaseErrors = []
    syncTimeStamps = timeStamps[states == 'sync']

    for triggerTime in triggerTimes:
        # Compute an array of the difference between timestamps and current trigger time
        timeDifferences = syncTimeStamps - triggerTime
        # Only interpolate phases for triggers which are followed by further timestamps
        # Then split into negative and positive parts to find the closest behind time/phase and closest ahead time/phase
        
        if not timeDifferences[-1] < 0:
            behindTime = timeDifferences[timeDifferences < 0][-1] + triggerTime
            aheadTime = timeDifferences[timeDifferences > 0][0] + triggerTime
            behindPhase = phases[syncTimeStamps == behindTime][0]
            aheadPhase = phases[syncTimeStamps == aheadTime][0]
        
        # Interpolate between the behind and ahead coordinate
        unwrappedPhase = np.interp(triggerTime, [behindTime, aheadTime], [behindPhase, aheadPhase])
        
        # Compute error between wrapped phase (unwrapped modulo 2pi) and current target phase
        phaseError = unwrappedPhase % (2 * np.pi) - targetPhases[syncTimeStamps == aheadTime][0]
        
        # Bring phase error into -pi, pi range
        if phaseError > np.pi:
            phaseError = phaseError - (2 * np.pi)
        elif phaseError < - np.pi:
            phaseError = phaseError + (2 * np.pi) 
            
        unwrappedPhases.append(unwrappedPhase), phaseErrors.append(phaseError)

    return unwrappedPhases, phaseErrors