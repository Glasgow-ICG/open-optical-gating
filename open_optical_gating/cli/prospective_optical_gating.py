""" Module for real-time phase matching of brightfield sequences.
    These codes are equivalent to the Objective-C codes in spim-interface."""

# Python imports
import numpy as np
from loguru import logger
import scipy.optimize

# Module imports
import j_py_sad_correlation as jps

# Local imports
from . import pixelarray as pa
from . import determine_reference_period as ref

# Rewrite this as a parent class and have two subclasses: linear predictor and Kalman predictor

class LinearPredictor:
    def __init__(self, predictor_settings):
        self.settings = predictor_settings
        self.mostRecentTriggerTime = -1000

    def target_and_barrier_updated(self, ref_seq_manager):
        """ This function should be called whenever the reference and barrier frames are updated.
            We generate a lookup table that enables us to identify how many frame phases to use
            for forward-prediction of heart phase, depending on what our current frame phase is.
            This is empirical code replicated from the spim-interface project.
            The concept of the 'barrier frame' is used: we do not fit backwards past this
            barrier frame. The barrier frame is identified empirically as a point in the
            reference sequence "between" heartbeats, where we may we see increased variability
            (variable pause between beats). Our forward-prediction is more reliable if
            we do not attempt to extend our linear fit backwards past that point
            of high variability in the heart cycle.
            """
        # JT TODO: this function is called when we acquire a new set of reference frames,
        # but it is not updated if the parameters such as barrierFrame, min/maxFramesForFit
        # are later altered by the user. It should be...
        
        # JT TODO: since it's not even that clear which class should manage this lookup,
        # I might be better off thinking about whether I can write it as reasonably compact logic
        # that does not require a lookup!
        
        # frames to consider based on reference point and no padding
        refFrameCountPadded = ref_seq_manager.ref_frames.shape[0]
        numToUseNoPadding = list(
            range(refFrameCountPadded - (2 * ref.numExtraRefFrames))
        )
        numToUseNoPadding = np.asarray(
            numToUseNoPadding[-int(ref_seq_manager.barrierFrameNum - ref.numExtraRefFrames - 1) :]
            + numToUseNoPadding[: -int(ref_seq_manager.barrierFrameNum - ref.numExtraRefFrames - 1)]
        )
        ones = np.ones(refFrameCountPadded, dtype=int)

        # Account for padding by setting extra frames equal to last/first unpadded number
        numToUsePadding = numToUseNoPadding[-1] * ones
        numToUsePadding[
            ref.numExtraRefFrames : refFrameCountPadded - ref.numExtraRefFrames
        ] = numToUseNoPadding

        # Consider min and max number of frames to use, as determined by our settings parameters.
        # This overrides any barrier frame considerations.
        numToUsePadding = np.maximum(
            numToUsePadding,
            self.settings["minFramesForFit"] * ones
        )
        numToUsePadding = np.minimum(
            numToUsePadding,
            self.settings["maxFramesForFit"] * ones
        )
        self.framesForPredictionLookup = numToUsePadding

    def predict_trigger_wait(self, full_frame_history, targetSyncPhase, frameInterval_s, fitBackToBarrier=True, framesForFit=None):
        """ Predict how long we need to wait until the heart is at the target phase we are triggering to.
            
            Parameters:
                full_frame_history      list        List of PixelArray objects, including appropriate metadata
                targetSyncPhase         float       Phase that we are supposed to be triggering at
                frameInterval_s         float       Expected time gap (in seconds) between brightfield frames
                fitBackToBarrier        bool        Should we use the "barrier frame" logic? (see determine_barrier_frames)
                framesForFit            int         Optional override value to fit to a specific number of frames from the past history.
                                                    Note that the main purpose of this parameter is for use when we recursively call ourselves
            Returns:
                Time delay (in seconds) between when the current frame was
                 acquired and when the trigger would need to be sent.
                Current heart period (in seconds), for information
            """

        # Deal with the barrier frame logic (if fitBackToBarrier is True):
        # Rather than fitting to a number of frames that depends on how far forward we are predicting,
        # fit to a number that depends on where in the cycle we are.
        # We try not to fit to the refractory period unless there really is no other data.
        # The intention of this is to fit to as much data as possible but only in the parts of the cycle
        # where the phase progression is highly predictable and linear with time.
        if framesForFit is None:
            if fitBackToBarrier:
                allowedToExtendNumberOfFittedPoints = False
                framesForFit = min(
                    self.framesForPredictionLookup[int(full_frame_history[-1].metadata["sad_min"])],
                    len(full_frame_history),
                )
                logger.debug("Consider {0} past frames for prediction;", framesForFit)
            else:
                framesForFit = self.settings["minFramesForFit"]
                allowedToExtendNumberOfFittedPoints = True

        if len(full_frame_history) < framesForFit:
            logger.debug("Fit failed due to too few frames")
            return -1
        
        # We need to look back through the recent frame history and pick out 'framesForFit' frames
        # to use in our linear fit. Note the care taken here to only pass those relevant frames
        # to get_metadata_from_list, so that the time taken inside that function doesn't get longer
        # and longer as the experiment continues (when our history is large)
        frame_history = pa.get_metadata_from_list(
                          full_frame_history[-framesForFit:], ["timestamp", "unwrapped_phase", "sad_min"]
                        )

        # Perform a linear fit to the past phases. We will use this for our forward-prediction
        logger.trace("Phase history times: {0}", frame_history[:, 0])
        logger.trace("Phase history phases: {0}", frame_history[:, 1])
        radsPerSec, alpha = np.polyfit(frame_history[:, 0], frame_history[:, 1], 1)
        logger.debug("Linear fit with intersect {0} and gradient {1}", alpha, radsPerSec)
        if radsPerSec < 0:
            logger.debug(
                "Linear fit to unwrapped phases is negative! This is a problem for the trigger prediction."
            )
        elif radsPerSec == 0:
            logger.debug(
                "Linear fit to unwrapped phases is zero! This will be a problem for prediction (divByZero)."
            )
        estHeartPeriod_s = 2*np.pi/radsPerSec

        # Use our linear fit to get a 'fitted' unwrapped phase for the latest frame
        # This should not rescue cases where, for some reason, the image-based
        # phase matching is erroneous.
        thisFramePhase = alpha + frame_history[-1, 0] * radsPerSec
        # Count how many total periods we have seen
        multiPhaseCounter = thisFramePhase // (2 * np.pi)
        # Determine how much of a cardiac cycle we have to wait till our target phase
        phaseToWait = targetSyncPhase + (multiPhaseCounter * 2 * np.pi) - thisFramePhase
        # c.f. function triggerAnticipationProcessing in SyncAnalyzer.mm
        # essentially this fixes for small backtracks in phase due to SAD imperfections.
        # If our computations so far suggest that our target phase is in the past, then
        # we add multiples of 2pi until we are targeting the same phase point in a future heartbeat.
        while phaseToWait < 0:
            phaseToWait += 2 * np.pi

        timeToWait_s = phaseToWait / radsPerSec
        timeToWait_s = max(timeToWait_s, 0.0)

        logger.debug(
            "Current time: {0};\tTime to wait: {1};",
            frame_history[-1, 0],
            timeToWait_s,
        )
        logger.debug(
            "Current phase: {0};\tPhase to wait: {1};", thisFramePhase, phaseToWait,
        )  
        logger.debug(
            "Target phase:{0};\tPredicted phase:{1};",
            targetSyncPhase + (multiPhaseCounter * 2 * np.pi),
            thisFramePhase + phaseToWait,
        )

        # Fixes sync error due to targetSyncPhase being 2pi greater than target phase (1e-3 is for floating point errors)
        if (
            thisFramePhase
            + phaseToWait
            - targetSyncPhase
            - (multiPhaseCounter * 2 * np.pi)
            > 2 * np.pi + 1e-3
        ):
            logger.warning(
                "Phase discrepency, trigger aborted. At {0} with wait {1} for target {2} [{3}]",
                thisFramePhase % (2 * np.pi),
                phaseToWait,
                targetSyncPhase,
                thisFramePhase
                + phaseToWait
                - targetSyncPhase
                - (multiPhaseCounter * 2 * np.pi),
            )
            timeToWait_s = 0.0

        # This logic catches cases where we are predicting a long way into the future using only a small number of datapoints.
        # That is likely to be error-prone, so (unless using the "barrier frame" logic) we may increase
        # the number of frames we use for prediction.
        # JT: note that this non-barrier-frame logic is not routinely used any more,
        # and could be improved a bit. I will leave all this for now, though, because I anticipate
        # it being replaced with better forward-prediction algorithms fairly soon anyway.
        if allowedToExtendNumberOfFittedPoints and timeToWait_s > (
            self.settings["extrapolationFactor"] * framesForFit * frameInterval_s
        ):
            extendedFramesForFit = framesForFit * 2
            if (
                extendedFramesForFit <= frameHistory.shape[0]
                and extendedFramesForFit <= self.settings["maxFramesForFit"]
            ):
                logger.debug("Repeating fit using more frames")
                #  Recurse, using a larger number of frames, to obtain an improved predicted time
                # (Note that if we get to this code branch, fitBackToBarrier will in fact definitely be False)
                timeToWait_s, estHeartPeriod_s = self.predict_trigger_wait(
                    frame_history, targetSyncPhase, frameInterval_s, fitBackToBarrier, extendedFramesForFit
                )

        # Return our prediction
        return timeToWait_s, estHeartPeriod_s

    def decide_whether_to_trigger(self, timestamp, timeToWait_s, frameInterval_s, estHeartPeriod_s):
        """ Potentially schedules a synchronization trigger for the fluorescence camera,
            based on the caller-supplied candidate trigger time described by timeToWait_s.
            We will do this if the trigger is due fairly soon in the future,
            and we are not confident we will have time to make an updated prediction
            based on the next incoming frame from the brightfield camera.
            
            Parameters:
                timestamp               float   Time associated with current frame (seconds)
                timeToWait_s            float   Time delay (in seconds) before trigger would need to be sent.
                frameInterval_s         float   Expected time gap (in seconds) between brightfield frames
                estHeartPeriod_s        float   Estimated period of heartbeat (in seconds)
            Returns:
                timeToWait_s            float   Updated time delay before trigger would need to be sent.
                                                 Note that this return value may be modified from its input value (see code below).
                sendTriggerReason       str     String indicates why a trigger for the fluorescence camera should be scheduled now,
                                                 for a time timeToWait_s into the future. Or None if no trigger should be sent
            """
        sendTriggerReason = None

        logger.debug(
            "Time to wait: {0} s; with latency: {1} s;",
            timeToWait_s,
            self.settings["prediction_latency_s"],
        )

        # The parameter 'prediction_latency_s' represents how much time we *expect* to need
        # between scheduling a trigger and actually being able to send it.
        # That influences whether we commit to this trigger time, or wait for an updated prediction based on the next brightfield frame due to arrive soon
        if self.mostRecentTriggerTime >= timestamp - estHeartPeriod_s / 2:
            # We have already sent a trigger on this heartbeat, so we consider that we are now making predictions for the *next* cycle.
            #
            # If we've done any triggering in the last half a cycle, don't trigger again.
            # This is quite different from JTs approach,
            # where he keeps track of which cycle we last triggered on.
            # JT note: the reason for my approach is because I may want to send multiple triggers at different heart phases
            # JT TODO: I may want to update this code, and/or incorporate that concept...
            logger.debug("Trigger already sent recently. Will not send another - extending the prediction coarsely to the next cycle")
            timeToWait_s += estHeartPeriod_s
        elif timeToWait_s < self.settings["prediction_latency_s"]:
            # Haven't sent a trigger on this heartbeat.
            # We may not have time, but we can give it a go and cross our fingers we schedule it in time
            logger.debug("Trigger is needed with short latency, but we may as well give it a shot...")
            sendTriggerReason = "panic"
        elif (timeToWait_s - (1.6 * frameInterval_s)) < self.settings["prediction_latency_s"]:
            # We don't expect to have time to wait for an updated prediction from the next frame... so schedule the trigger now!
            # Note that the 1.6 multiplier is an empirical constant related to how much
            # frame-to-frame variability we expect in the phase rate.
            # Ideally it would depend on the  actual observed variability in the time estimates as successive frame data is received
            logger.debug("Schedule trigger because target time is coming up soon")
            sendTriggerReason = "standard"
        else:
            # We expect to have time to wait for an updated prediction, so we do nothing for now.
            logger.debug("No trigger - we reckon we can wait for next frame")
            pass

        if (sendTriggerReason is not None):
            logger.debug("Trigger scheduled to be sent, updating `mostRecentTriggerTime` to {0}+{1}.", timestamp, timeToWait_s)
            self.mostRecentTriggerTime = timestamp + timeToWait_s

        return timeToWait_s, sendTriggerReason
