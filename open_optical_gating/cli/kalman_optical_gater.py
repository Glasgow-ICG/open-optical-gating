import json
from loguru import logger
import sys, os, time, argparse, glob, warnings, platform
import numpy as np
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
    
    def predict_trigger_wait(self, phase, targetSyncPhase, frameInterval_s):
        raise NotImplementedError("Subclasses must override this function")
    
class LinearPredictor(PredictorBase):
    def __init__(self, settings) -> None:
        super().__init__(settings)

    def predict_trigger_wait(self, phase, targetSyncPhase, frameInterval_s):
        phase_time = targetSyncPhase - phase
    
class KalmanPredictor(PredictorBase):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.initialised = False

    def predict_trigger_wait(self, phase, targetSyncPhase, frameInterval_s):
        # We need to initialise the KF with an initial state estimate. For this we use the first phase estimate from
        # open optical gating. If the Kalman filter is already initialised we run the KF.
        # NOTE: Currently, we don't provide an estimate for our phase velocity. Possible choices include,
        # calculating the expected velocity from our known brightfield framerate or using the first two phase
        # estimates from OOG
        if self.initialised == False:
            # Initialise the using the current phase and velocity estimate
            # TODO: Replace the number 75 with the correct framerate
            # We can get this from the example_data_settings.json
            x_0 = np.array([0, 10])
            P_0 = np.diag([100, 100])
            q = 2.08
            R = 1
            self.kf = KalmanFilter.constant_velocity_2(self.settings, frameInterval_s, q, R, x_0, P_0)
            # May in future need the reference period for estimating the initial delta phase
            self.kf.initialise(np.array([phase, 1]), self.kf.P)
            self.initialised = True
        else:
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

    def predict_trigger_wait(self, phase, targetSyncPhase, frameInterval_s):
        # We need to initialise the KF with an initial state estimate. For this we use the first phase estimate from
        # open optical gating. If the Kalman filter is already initialised we run the KF.
        # NOTE: Currently, we don't provide an estimate for our phase velocity. Possible choices include,
        # calculating the expected velocity from our known brightfield framerate or using the first two phase
        # estimates from OOG
        if self.initialised == False:
            # Initialise the using the current phase and velocity estimate
            # TODO: Replace the number 75 with the correct framerate
            # We can get this from the example_data_settings.json
            x_0 = np.array([0, 10])
            P_0 = np.diag([100, 100])
            q = 2.08
            R = 1
            self.kf1 = KalmanFilter.constant_velocity_2(self.settings, frameInterval_s, q / 100, R, x_0, P_0)
            self.kf2 = KalmanFilter.constant_velocity_2(self.settings, frameInterval_s, q * 100, R, x_0, P_0)
            self.IMM = IMM([self.kf1, self.kf2]), np.array([0.5, 0.5]), np.array([[0.97, 0.03],[0.03, 0.97]])
            # May in future need the reference period for estimating the initial delta phase
            self.kf.initialise(np.array([phase, 1]), self.kf.P)
            self.initialised = True
        else:
            # Run the KF
            self.kf.predict()
            self.kf.update(phase)

        # This code attempts to predict how long we need to wait until the next trigger by estimating the
        # phase remaining and KF estimate of phase velocity.
        timeToWait_s, estHeartPeriod_s = KalmanFilter.get_time_til_phase(self.imm.x, targetSyncPhase)

        # Return the remaining time and the estimated heart period
        return timeToWait_s, estHeartPeriod_s, None

        

def initialise_predictor(settings):
    # Initialise our predictor
    if settings["prediction"]["prediction_method"] == "linear":
        # Linear predictor
        predictor = LinearPredictor(settings["prediction"]["linear"])
    elif settings["prediction"]["prediction_method"] == "kalman":
        # Kalman filter predictor
        dt = 1 / settings["brightfield"]["brightfield_framerate"]
        predictor = KalmanPredictor(settings["prediction"]["kalman"], dt)
    elif settings["prediction"]["prediction_method"] == "IMM":
        # IMM Kalman filter predictor
        dt = 1 / settings["brightfield"]["brightfield_framerate"]
        predictor = IMMPredictor(settings["prediction"]["IMM"], dt)
    else:
        raise NotImplementedError("Unknown prediction method '{0}'".format(settings["prediction"]["prediction_method"]))
    
    return predictor


if __name__ == "__main__":
    settings = load_settings("./optical_gating_data/example_data_settings.json")
    predictor = initialise_predictor(settings)

    phases = np.linspace(0, 100, 1000)

    for phase in phases:
        predictor.predict_trigger_wait(phase, 30, 0.1)