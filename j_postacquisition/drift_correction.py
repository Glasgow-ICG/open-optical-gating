from shifts import *

def CorrectForDrift(resampledImageSections, numSamplesPerPeriod, maxDelta=1, inset=10):
    # This function corrects for any lateral movement that may have occurred between two sequences,
    # returning a list of the shifts that should be applied to each of the sequences we were passed.
    #
    # This lateral motion can occur due to:
    # 1. Genuine sample motion
    # 2. Unintended lateral image shift during focus correction
    # 3. Expected lateral image shift during a side-imaged z scan
    # Note though that I intend to at least make a good attempt at correcting the lateral image shift in #3
    # before we get to this point, at the stage when we are loading the images.
    #
    # At the moment I hard-code a maximum drift offset (the variable 'inset').
    # This does pose some restrictions for the code - we should be aware that it may not be able to cope with drift over the full range of a z scan
    # Note also that, as written at the moment, this definitely rules out processing size-scan sequences without a preprocessing step based on the observed z coords.
    #
    # Important note: this code currently compares everything against the FIRST sequence we are provided with.
    # This may be sub-optimal in the case of a long focus-corrected scan (where other things like magnification may change slightly),
    # and if we use this function in any way for long-term datasets where things may change significantly over the course of the dataset.
    # And, for that matter, for any dataset where there is a time gap in the sequences we are provided with, since we assume the drift
    # does not undergo large step changes.
    #
    #
    # Function works by comparing each sequence against the first one, and trying a small set of different drift values
    # to see which gives the lowest SAD correlation (having previously estimated the sync between the two sequences, to line them up correctly).
    # That optimum drift value is provided as the answer.
    (iSize, jSize) = resampledImageSections[0][0].image.shape
    
    # TODO: for side-scan drift, I need to make an estimated correction at the image-loading stage (based on known z) on a per-image basis,
    # and only actually load a sub-window of the images (a sub-window that is expected to be approximately drift-corrected)
    
    sequenceDrifts = [(0,0)]
    maxDriftObserved = (0, 0)
    # inset needs to be enough to cope with all possible drift offsets we may encounter (or we need to limit the maximum drift we correct for)
    # maxDelta is the maximum change in drift we expect between ADJACENT sequences, but of course the drift may accumulate over the course of
    # the full image sequence we have been provided to work on. For sequences I have looked at so far, I have found that maxDelta=1 keeps up ok.
    m = maxDelta # shorthand
    directScores = []
    for n in tqdm(range(1, len(resampledImageSections)), desc='drift correction'):
        # Compare section i with section 0
        seqA = resampledImageSections[0]
        seqB = resampledImageSections[n]
        lastIDrift = min(max(sequenceDrifts[n-1][0], -(inset-m)), inset-m)
        lastJDrift = min(max(sequenceDrifts[n-1][1], -(inset-m)), inset-m)
        
        # First we need to know what the best phase shift is for this combination
        scores = ShiftScoresForSequences(seqA, seqB, numSamplesPerPeriod, window1=MakeOffsetWindow(iSize, jSize, inset, 0, 0), window2=MakeOffsetWindow(iSize, jSize, inset, 0, 0))
        bestShift = np.argmin(scores)
        bestDirectScore = np.min(scores)
        directScores.append(bestDirectScore)
        # Now consider whether a spatial shift might help
        scoreArray = np.zeros((2*m+1, 2*m+1))
        diArray = np.tile(np.arange(0, 2*m+1) - m + lastIDrift, (2*m+1, 1)).transpose()
        djArray = np.tile(np.arange(0, 2*m+1) - m + lastJDrift, (2*m+1, 1))
        for i in range(2*m+1):
            for j in range(2*m+1):
                di = diArray[i,j]
                dj = djArray[i,j]
                window1 = MakeOffsetWindow(iSize, jSize, inset, 0, 0)
                window2 = MakeOffsetWindow(iSize, jSize, inset, di, dj)
                score = ScoreSequenceShift(seqA, seqB, bestShift, numSamplesPerPeriod, window1=window1, window2=window2)
                scoreArray[i,j] = score
        di = diArray.flat[np.argmin(scoreArray)]
        dj = djArray.flat[np.argmin(scoreArray)]
        
        if False:
            # Diagnostic check to see whether the same sequence shift still applies,
            # or whether drift correction has changed the best shift value
            scores2 = ShiftScoresForSequencesWithDrift(seqA, seqB, (di, dj), inset, numSamplesPerPeriod)
            bestShift2 = np.argmin(scores2)
            bestDirectScore2 = np.min(scores2)
            # bestDirectScore - value at minimum position in the intitial shift scores
            # np.min(scoreArray) - for that phase shift, minimum score we found when trying different drifts
            # bestDirectScore2 - value at minimum position in the new shift scores when using that drift
            #print (n, seqB[0].frameIndex, di, dj, bestShift, bestShift2, bestDirectScore, np.min(scoreArray), bestDirectScore2)
                
        maxDriftObserved = (max(maxDriftObserved[0], abs(di)), max(maxDriftObserved[1], abs(dj)))
        sequenceDrifts = sequenceDrifts + [(di, dj)]

    if ((maxDriftObserved[0] == inset) or (maxDriftObserved[1] == inset)):
        print ('WARNING - drift correction saturated:', maxDriftObserved)
    return sequenceDrifts
