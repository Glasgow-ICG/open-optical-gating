import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, sin

def SolveForShifts(shifts, numSequences, knownPhaseIndex, knownPhase):
    # Build a matrix/vector describing the system of equations Mx = a
    # This expects an input of 'shifts' consisting of triplets of (seq1Index, seq2Index, shift)
    # and an integer giving the number of sequences
    # (maximum value appearing for sequence index should be numSequences-1)
    # Note that this function forces the absolute phase of the first sequence
    # to be equal to phaseForFirstSequence.
    M = np.zeros((len(shifts)+1, numSequences))
    a = np.zeros(len(shifts)+1)
    w = a.copy()
    for n in range(len(shifts)):
        (i, j, shift, score) = shifts[n]
        M[n, i] = -1
        M[n, j] = 1
        a[n] = shift
        w[n] = 1.0 / score
    M[len(shifts), knownPhaseIndex] = 1
    a[len(shifts)] = knownPhase
    w[len(shifts)] = 1

    # I tried Liebling's weighted least squares formula and couldn't seem to get it to work.
    # This is one from http://stackoverflow.com/questions/19624997/understanding-scipys-least-square-function-with-irls
    Mw = M * np.sqrt(w[:,np.newaxis])
    aw = a * np.sqrt(w)
    (selfConsistentShifts, residuals, rank, s) = np.linalg.lstsq(Mw, aw, rcond=None)
    return (selfConsistentShifts, residuals)

def SolveWithMaxRange(shifts, numSequences, maxRange, knownPhaseIndex, knownPhase, log=True):
    shiftsToUse = []
    for n in range(len(shifts)):
        (i, j, shift, score) = shifts[n]
        if (j <= i+maxRange):
            shiftsToUse.append(shifts[n])
    if log:
        print('Solving using', len(shiftsToUse), 'of', len(shifts), 'constraints (max range', maxRange, ')')
    return SolveForShifts(shiftsToUse, numSequences, knownPhaseIndex, knownPhase)

def AdjustShiftsToMatchSolution(shifts, partialShiftSolution, periods, warnUpTo=65536):
    # Now adjust the longer-distance shifts so they match our initial solution
    adjustedShifts = []
    if type(periods) is int or len(periods)==1:
        period = periods
    for n in range(len(shifts)):
        (i, j, shift, score) = shifts[n]
        if type(periods) is list and len(periods)>1:
            period = periods[i]
        expectedWrappedShift = (partialShiftSolution[j] - partialShiftSolution[i]) % period
        periodPart = (partialShiftSolution[j] - partialShiftSolution[i]) - expectedWrappedShift
        discrepancy = expectedWrappedShift - shift
        if (abs(discrepancy) < (period / 4.0)):
            # If discrepancy is small (positive or negative)
            # then add an appropriate number of periods to make it work
            adjustedShift = shift + periodPart
        elif (abs(discrepancy) > (3 * period / 4.0)):
            # Values look consistent, but cross a phase boundary
            if (expectedWrappedShift < shift):
                adjustedShift = shift + (periodPart - period)
            else:
                adjustedShift = shift + (periodPart + period)
        else:
            if (j-i <= warnUpTo):
                print ('major discrepancy between approx expected value', expectedWrappedShift, 'and actual value', shift, 'for', (i, j), '(distance', j-i, 'score', score, ')')
            # Exclude this shift because we aren't sure how to adjust it (yet)
            # Hopefully things may become clearer as we refine our estimated overall solution
            adjustedShift = None

        if (adjustedShift is not None):
            adjustedShifts.append((i, j, adjustedShift, score))
    return adjustedShifts

def MakeShiftsSelfConsistent(shifts, numSequences, period, knownPhaseIndex=0, knownPhase=0, log=True):
    # Given a set of what we think are the optimum relative time-shifts between different sequences
    # (both adjacent sequences, and some that are further apart), work out a global self-consistent solution.
    # The longer jumps serve to protect against gradual accumulation of random error in the absolute global phase,
    # which would creep in if we only ever considered the relative shifts of adjacent sequences.

    # First solve just using the shifts between adjacent slices (no phase wrapping)
    # TODO: add a more comprehensive comment here explaining the modulo-2pi issues that make the
    # shift problem a little bit awkward.
    (adjacentShiftSolution, adjacentResiduals) = SolveWithMaxRange(shifts, numSequences, 1, knownPhaseIndex, knownPhase, log)
    # Adjust the longer shifts to be consistent with the adjacent shift values
    # Don't warn about long-distance discrepancies, because those are fairly inevitable initially
    adjacentShifts = AdjustShiftsToMatchSolution(shifts, adjacentShiftSolution, period, warnUpTo=64)

    if log:
        print('Done first stage')

    # Now look for a solution that satisfies longer-range shifts as well.
    # If necessary, we could make a new adjustment of the shifts and repeat.
    # On a subsequent iteration we would have an improved estimate that might help us
    # decide which way to adjust long-range shifts that were initially unclear
    adjustedShifts = list(adjacentShifts)
    for r in [32, 128, 512, 2048]:
        (shiftSolution, residuals) = SolveWithMaxRange(adjustedShifts, numSequences, r, knownPhaseIndex, knownPhase, log)
        adjustedShifts = AdjustShiftsToMatchSolution(shifts, shiftSolution, period)

    return (shiftSolution, adjustedShifts, adjacentShiftSolution, residuals, adjacentResiduals)
