import numpy as np
import matplotlib.pyplot as plt

def AssignPhaseToFourPiImageSection(phaseStart, imageSection):
    # Assign phases to a sequence of (resample) images that should represent exactly two full phases.
    # This means that the final image is just *before* the 4pi point, of course.
    times = []
    phases = []
    for i in range(len(imageSection)):
        imageSection[i].phase = (phaseStart + (i * 4*np.pi / len(imageSection))) % (2*np.pi)
        times.append(imageSection[i].timestamp)
        phases.append(imageSection[i].phase)
    # Optional: print out the timestamp and phase for the start of this image section, and the timestamp for the point 4pi further on.
    #print(imageSection[0].timestamp, imageSection[0].phase, 2*imageSection[-1].timestamp -imageSection[-2].timestamp)
    return (times, phases)
        
def DefinePhaseForSequence(resampledImageSections, shiftSolution, numSamplesPerPeriod, plotIt=False, interpolationDistanceBetweenSequences=0):
    # Given a shiftSolution that describes how much each image section should be shifted by in order to
    # synchronize all the image sections relative to each other, we can deduce what the heart phase is
    # at each point in time
    # We return a pair of arrays representing that known time/phase mapping information.
    # That could then be used to phase-stamp a different image channel, if we want.
    assert(len(resampledImageSections) == len(shiftSolution))
    knownTimes = []
    knownPhases = []
    for i in range(len(resampledImageSections)):
        # A positive shift means that the second sequence matches
        # if we consider phase 0 to be at a point at distance 'shift' into the second sequence
        # In other words, the second sequence starts at the NEGATIVE phase equivalent to 'shift'
        (theseTimestamps, thesePhases) = AssignPhaseToFourPiImageSection(-shiftSolution[i] / numSamplesPerPeriod * 2 * np.pi, resampledImageSections[i])
        
        # If this is not the first image section, we may optionally want to print out information on the discontinuities between sections
        if False and (i > 0):
            # Extrapolate linearly from last two knownTimes, and see how that phase differs from the first one we have for the current image section
            thisDelta = theseTimestamps[0] - knownTimes[-1]
            prevDelta = knownTimes[-1] - knownTimes[-2]
            prevDeltaPhase = (knownPhases[-1] - knownPhases[-2]) % (2*np.pi)
            extrapolatedPhase = knownPhases[-1] + prevDeltaPhase * (thisDelta / prevDelta)
            discontinuityInPhase = (extrapolatedPhase - thesePhases[0]) % (2*np.pi)
            if (discontinuityInPhase > np.pi):
                # Go for slightly negative rather than almost 2pi
                discontinuityInPhase = discontinuityInPhase - 2.0*np.pi
            # The discontinuity in time is a bit more fiddly I think. You have two lines with different slopes,
            # so you have to decide at what point you want to measure the discontinuity.
            # For the phase discontinuity, I have extrapolated the first slope forwards to the start of the second one.
            # I suppose the logical equivalent would be to measure the (horizontal) TIME gap between that starting point
            # and the straight line of the previous sequence.
            # So, the discontinuity in phase can be translated to a time using the gradient of the first sequence
            discontinuityInTime = discontinuityInPhase / prevDeltaPhase * prevDelta
            print('DISC', discontinuityInPhase, discontinuityInTime)
        
        # We may wish to force unknown phases for anything outside the known range, or may be willing to interpolate across short gaps
        permittedGap = (theseTimestamps[1] - theseTimestamps[0]) * interpolationDistanceBetweenSequences
        nanBefore = False
        if (i == 0):
            # Force unknown phase for anything before the start of the range
            nanBefore = True
        elif ((theseTimestamps[0] - knownTimes[-1]) > permittedGap):
            # Force unknown phase in this gap, which is too large to be permitted
            nanBefore = True
        
        if (nanBefore):
            # Force unknown phases for anything outside the known range
            knownTimes = knownTimes + [theseTimestamps[0]-1e-6] + theseTimestamps
            knownPhases = knownPhases + [np.nan] + thesePhases
        else:
            knownTimes = knownTimes + theseTimestamps
            knownPhases = knownPhases + thesePhases

    # Force unknown phase for anything beyond the end of the range
    knownTimes = knownTimes + [knownTimes[-1]+1e-6]
    knownPhases = knownPhases + [np.nan]
    
    if (plotIt):
        plt.plot(knownTimes, knownPhases, '.')
    return (np.array(knownTimes), np.array(knownPhases))
    
