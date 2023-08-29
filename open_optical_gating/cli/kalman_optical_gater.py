# Python imports
import numpy as np
from loguru import logger
import scipy.optimize
import json

# Module imports
import j_py_sad_correlation as jps

# Local imports
from . import pixelarray as pa
from . import determine_reference_period as ref
from .kalman_filter import KalmanFilter
from .kalman_filter import InteractingMultipleModelFilter as IMM

def load_settings(settings_file_path):
    '''
        Load the settings.json file
    '''

    # Load the file as a settings file
    logger.success("Loading settings file {0}...".format(settings_file_path))
    try:
        with open(settings_file_path) as data_file:
            settings = json.load(data_file)
    except FileNotFoundError:
        logger.exception("Could not find the specified settings file.")

    return settings

class PredictorBase():
    def __init__(self, settings) -> None:
        self.settings = settings
    
    def predict_trigger_wait(self, phase, timestamp, targetSyncPhase, frameInterval_s, reference_period):
        raise NotImplementedError("Subclasses must override this function")
    
class LinearPredictor(PredictorBase):
    def __init__(self, settings) -> None:
        super().__init__(settings)

        self.phase_history = []
        self.timestamp_history = []

    def predict_trigger_wait(self, phase, timestamp, targetSyncPhase, frameInterval_s, reference_period):
        fitBackToBarrier = True

        self.phase_history.append(phase)
        self.timestamp_history.append(timestamp)

        if len(self.phase_history) > self.settings["general"]["frame_buffer_length"]:
            del self.phase_history[0]
        if len(self.timestamp_history) > self.settings["general"]["frame_buffer_length"]:
            del self.timestamp_history[0]
        
        nFramesForFit = self.settings["prediction"]["linear"]["maxFramesForFit"]

        if len(self.phase_history) < nFramesForFit:
            logger.debug("Fit failed due to too few frames")
            return -1, -1, -1
        
        # We need to look back through the recent frame history and pick out 'framesForFit' frames
        # to use in our linear fit. Note the care taken here to only pass those relevant frames
        # to get_metadata_from_list, so that the time taken inside that function doesn't get longer
        # and longer as the experiment continues (when our history is large)
        phasesForFit = self.phase_history[-nFramesForFit:]
        timestampsForFit = self.timestamp_history[-nFramesForFit:]
        
        # Test for gaps in the timestamps of the frames we are using.
        # Really we are looking to detect large gaps that indicate we stopped acquiring (e.g. while doing a LTU),
        # because that means the phase unwrapping is probably not correct across the time gap.
        # But it's probably also no bad thing that this will detect any serious jitted in frame arrival times
        # (which would make me nervous when fitting)
        ts = np.array(self.timestamp_history)
        if (len(ts) > 1):
            tsdiffs = ts[1:] - ts[:-1]
            if (np.max(tsdiffs) > np.min(tsdiffs) * 2.5):
                logger.info("Fit failed due to gap in frame history of time {0} ({1}, {2}, {3})", np.max(tsdiffs), len(tsdiffs), np.min(tsdiffs), np.median(tsdiffs))
                return -1, -1, -1
        
        # Perform a linear fit to the past phases. We will use this for our forward-prediction
        logger.trace("Phase history times: {0}", timestampsForFit)
        logger.trace("Phase history phases: {0}", phasesForFit)

        radsPerSec, alpha = np.polyfit(timestampsForFit, phasesForFit, 1)
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
        thisFramePhase = alpha + self.timestamp_history[-1] * radsPerSec
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
            self.phase_history[-1],
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
        """if allowedToExtendNumberOfFittedPoints and timeToWait_s > (
            self.settings["extrapolationFactor"] * framesForFit * frameInterval_s
        ):
            extendedFramesForFit = framesForFit * 2
            if (
                extendedFramesForFit <= frame_history.shape[0]
                and extendedFramesForFit <= self.settings["maxFramesForFit"]
            ):
                logger.debug("Repeating fit using more frames")
                #  Recurse, using a larger number of frames, to obtain an improved predicted time
                # (Note that if we get to this code branch, fitBackToBarrier will in fact definitely be False)
                timeToWait_s, estHeartPeriod_s = self.predict_trigger_wait(
                    frame_history, targetSyncPhase, frameInterval_s, fitBackToBarrier, extendedFramesForFit
                )"""

        # Add wait time to metadata
        """thisFrameMetadata = full_frame_history[-1].metadata
        thisFrameMetadata["states"] = np.array([alpha, radsPerSec])
        thisFrameMetadata["wait_times"] = timeToWait_s"""

        # Return our prediction
        return timeToWait_s, estHeartPeriod_s, timestamp
    
class KalmanPredictor(PredictorBase):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.initialised = False
        print("Kalman")

    def predict_trigger_wait(self, phase, timestamp, targetSyncPhase, frameInterval_s, reference_period):
        # We need to initialise the KF with an initial state estimate. For this we use the first phase estimate from
        # open optical gating. If the Kalman filter is already initialised we run the KF.
        # NOTE: Currently, we don't provide an estimate for our phase velocity. Possible choices include,
        # calculating the expected velocity from our known brightfield framerate or using the first two phase
        # estimates from OOG
        if self.initialised == False:
            # Initialise our KF
            x_0 = np.array([0, 10])
            P_0 = np.diag([1, 100])
            q = 2.08
            R = 1
            self.kf = KalmanFilter.constant_velocity_2(self.settings["prediction"]["kalman"], frameInterval_s, q, R, x_0, P_0)
            self.kf.initialise(np.array([phase, (2 * np.pi / reference_period) / frameInterval_s]), self.kf.P)
            self.state = "initialise_kf_delta_phase"

            self.initialised = True

            return -1, -1, -1
        elif self.initialised == True:
            # Run the KF
            self.kf.predict()
            self.kf.update(phase)

        # This code attempts to predict how long we need to wait until the next trigger by estimating the
        # phase remaining and KF estimate of phase velocity.
        timeToWait_s, estHeartPeriod_s = KalmanFilter.get_time_til_phase(self.kf.x, targetSyncPhase)

        # Return the remaining time and the estimated heart period
        return timeToWait_s, estHeartPeriod_s, None

class IMMPredictor(PredictorBase):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.initialised = False
        print("IMM")

    def predict_trigger_wait(self, phase, timestamp, targetSyncPhase, frameInterval_s, reference_period):
        # We need to initialise the KF with an initial state estimate. For this we use the first phase estimate from
        # open optical gating. If the Kalman filter is already initialised we run the KF.
        if self.initialised == False:
            # Initialise the using the current phase and velocity estimate
            # TODO: Replace the number 75 with the correct framerate
            # We can get this from the example_data_settings.json
            x_0 = np.array([phase, (2 * np.pi / reference_period) / frameInterval_s])
            P_0 = np.diag([1, 100])
            q = 2.08
            R = 1
            mu = np.array([0.5, 0.5])
            M = np.array([[0.97, 0.03],[0.03, 0.97]])
            self.kf1 = KalmanFilter.constant_velocity_2(self.settings["prediction"]["IMM"], frameInterval_s, q / 10, R, x_0, P_0)
            self.kf2 = KalmanFilter.constant_velocity_2(self.settings["prediction"]["IMM"], frameInterval_s, q * 10, R, x_0, P_0)
            models = [self.kf1, self.kf2]
            self.imm = IMM(models, mu, M)
            self.initialised = True
            return -1, -1, -1
        else:
            # Run the KF
            self.imm.predict()
            self.imm.update(phase)

        # This code attempts to predict how long we need to wait until the next trigger by estimating the
        # phase remaining and KF estimate of phase velocity.
        timeToWait_s, estHeartPeriod_s = KalmanFilter.get_time_til_phase(self.imm.x, targetSyncPhase)

        # Return the remaining time and the estimated heart period
        return timeToWait_s, estHeartPeriod_s, None

        

def initialise_predictor(settings):
    # Initialise our predictor
    if settings["prediction"]["prediction_method"] == "linear":
        # Linear predictor
        predictor = LinearPredictor(settings)
    elif settings["prediction"]["prediction_method"] == "kalman":
        # Kalman filter predictor
        predictor = KalmanPredictor(settings)
    elif settings["prediction"]["prediction_method"] == "IMM":
        # IMM Kalman filter predictor
        predictor = IMMPredictor(settings)
    else:
        raise NotImplementedError("Unknown prediction method '{0}'".format(settings["prediction"]["prediction_method"]))
    
    return predictor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Load the settings and setup our predictor
    settings = load_settings("./optical_gating_data/example_data_settings.json")
    predictor = initialise_predictor(settings)

    # Set the reference period, this will 
    reference_period = 38

    # Set our target sync phase and frame interval
    targetSyncPhase = 0
    frameInterval_s =  1 / settings["brightfield"]["brightfield_framerate"]

    # Generate an array of test phases
    phases = np.arange(0, 1000 * 2 * np.pi / reference_period, 2 * np.pi / reference_period)
    #phases += np.random.normal(0, 0.1, len(phases))
    timestamps = np.arange(0, 1000 * frameInterval_s, frameInterval_s)
    trigger_times_kf = []

    # Loop through our phases and get our time til trigger
    for i in range(len(phases)):
        trigger_times_kf.append(predictor.predict_trigger_wait(phases[i], timestamps[i], targetSyncPhase, frameInterval_s, reference_period)[0])

    trigger_times = np.array(trigger_times_kf)

    plt.plot(timestamps, trigger_times)
    plt.show()