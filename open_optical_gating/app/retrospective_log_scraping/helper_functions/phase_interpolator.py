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
        triggerPhases = array of interpolated unwrapped phase at each sent trigger time
        triggerPhaseErrors = array of phase error (interpolated phase - target phase) at each trigger time
    """
    
    triggerPhases = []
    triggerPhaseErrors = []
    syncTimeStamps = timeStamps[states == 'sync']

    for triggerTime in triggerTimes:
        # Compute an array of the difference between timestamps and current trigger time
        timeDifferences = syncTimeStamps - triggerTime
        # Only interpolate phases for triggers which are followed by further timestamps
        # Then split into negative and positive parts to find the closest behind time/phase and closest ahead time/phase
        behindTime = timeDifferences[timeDifferences < 0][-1] + triggerTime
        behindPhase = phases[syncTimeStamps == behindTime][0]
        if not timeDifferences[-1] < 0:
            aheadTime = timeDifferences[timeDifferences > 0][0] + triggerTime
            aheadPhase = phases[syncTimeStamps == aheadTime][0]
            
            if aheadTime - behindTime < 0.1:
                triggerPhase = np.interp(triggerTime, [behindTime, aheadTime], [behindPhase, aheadPhase])
                triggerPhaseError = triggerPhase % (2 * np.pi) - targetPhases[syncTimeStamps == aheadTime][0]
                if triggerPhaseError > np.pi:
                    triggerPhaseError = triggerPhaseError - (2 * np.pi)
                elif triggerPhaseError < - np.pi:
                    triggerPhaseError = triggerPhaseError + (2 * np.pi) 
                triggerPhases.append(triggerPhase), triggerPhaseErrors.append(triggerPhaseError)
                
            else:
                triggerPhases.append(behindPhase), triggerPhaseErrors.append(0)
        else:
                triggerPhases.append(behindPhase), triggerPhaseErrors.append(0)

    return triggerPhases, triggerPhaseErrors
