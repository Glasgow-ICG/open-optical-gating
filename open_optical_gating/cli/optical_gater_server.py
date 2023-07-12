"""Parent Open Optical Gating Class"""

# Python imports
import sys, time

# Module imports
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Optical Gating Alignment module
import optical_gating_alignment.optical_gating_alignment as oga

# Local imports
from . import pixelarray as pa
from . import determine_reference_period as ref
from . import prospective_optical_gating as pog

# Log information about the environment we are running in.
# This module list should match those specified in pyproject.toml
import importlib
version_dict = {"python": sys.version}
for mod in ["optical_gating_alignment", "j_py_sad_correlation", "loguru", "tqdm", "serial", "flask",
            "tifffile", "skimage", "scipy", "matplotlib", "numpy", "numba", "picamera", "fastpins", "pybase64"]:
    try:
        m = importlib.import_module(mod)
        version_dict[mod] = m.__version__
    except:
        version_dict[mod] = "<unavailable>"
logger.debug("Running in module environment: {0}", version_dict)

# TODO create a time-stamped copy of the settings file when the optical gater class is initialised
# TODO write the git version to the log file

# JT TODO: move update_criterion into LTU code, storing criteria in a dictionary, and resetting them when the LTU actually occurs

# JT TODO: at the moment we rely on self.settings["brightfield_framerate"] as our expectation of how often frames are arriving.
# (The relevant variable is frameInterval_s)
# It might be better to determine this empirically, and/or to be aware if the framerate changes significantly.

# JT TODO: I don't think barrier frames are being implemented properly.
# There is a call to determine_barrier_frames, but I don't think the *value* for the barrier frame parameter is ever computed, is it?
# It certainly isn't when using existing reference frames. This seems like an important missing bit of code.
# I think it just defaults to 0 when the settings are initialised, and stays that way.

class OpticalGater:
    """ Base optical gating class - includes no hardware features beyond
        placeholder functions for incoming brightfield source.

        This function carries out the logic required for adaptive prospective
        optical gating using an incoming data source and resulting in the
        determination of phase-locked trigger times.

        The OpticalGater depends on an internal state (self.state), which
        has the following modes:
            "reset" - re-initialise (clears for "determine" mode)
            "determine" - get period mode (requires user input; needed for "sync")
            "sync" - run prospective gating mode (phase-locked triggering)
            "adapt" - adaptive mode (update period but maintain phase-lock with previous period)
    """

    def __init__(self, settings = None, ref_frames = None, ref_frame_period = None):
        """Function inputs:
            settings - a dictionary of settings (see optical_gating_data/json_format_description.md)
        """
        # store the whole settings dict
        # we occasionally store some of this information elsewhere too
        # that's not ideal but works for now
        self.settings = settings
        
        # Mark this run with a unique identifier by adding to a counter -> helps with reference sequence collection
        if not "log_counter" in self.settings["general"]:
            self.settings["general"]["log_counter"] = 0
        self.settings["general"]["log_counter"] += 1
        
        # Set up logging, now receives level from settings
        logger.remove()
        logger.remove()
        logger.add("user_log_folder/oog_{time}.log", level = settings["general"]["log_level"], format = "{time:YYYY-MM-DD | HH:mm:ss:SSSSS} | {level} | {module}:{name}:{function}:{line} --- {message}")
        logger.add(sys.stderr, level = settings["general"]["log_level"])
        logger.enable("open_optical_gating")
        
        logger.debug("Instantiated OpticalGater with settings: {0}", settings)
        logger.debug("Initialising internal parameters...")
        self.initialise_internal_parameters()
        
        # Change "state" to prepare for a full reset
        self.state = "prep_full_reset"
        
        # Check if the user has supplied reference frames 
        # Only really applicable in file_optical_gater
        if ref_frames is not None:
            self.supplied_ref_frames = ref_frames
            self.supplied_ref_frame_period = ref_frame_period
    
    def initialise_internal_parameters(self):
        """Defines all internal parameters not already initialised"""
        self.frame_history = []
        self.frames_to_save = []
        self.slow_action_occurred = None
        self.latestTriggerPredictFrame = None
        
        # Define parameters for reference sequence update requirements
        self.trigger_num = 0 # Number of triggers sent since last reference update
        self.timelapse_trigger_num = 0 # Number of triggers sent since last timelapse reference update
        self.trigger_num_total = 0 # Total number of triggers send
        self.frame_num = 0 # Number of frames received since last reference update
        self.frame_num_total = 0 # Total number of frames received
        
        # Object timestamp for RPi time
        self.currentTimeStamp = 0
        self.mostRecentTriggerTime = 0
        # Flag for interrupting the program's running
        self.stop = False
        
    def run_server(self):
        """ Run the OpticalGater server, acting on the supplied frame images.
            run_and_analyze_until_stopped() is implemented by subclasses, which call back into
            analyze_pixelarray() with the appropriate frame data to be analyzed
        """
        self.run_and_analyze_until_stopped()
        if ((self.stop == "out-of-frames") and
            (self.state == "determine") and
            (self.aligner2.sequence_history is None)
        ):
            raise RuntimeError("Ran out of frames without ever managing to establish a period")

    def run_and_analyze_until_stopped(self):
        """ Subclasses must override this function to obtain images and pass them to analyze_pixelarray().
        """
        logger.error("Subclasses must override this function")
        raise NotImplementedError("Subclasses must override this function")

    def analyze_pixelarray(self, pixelArray):
        """ Method to analyse each frame as they are captured by the camera.
            Note that this analysis must take place fast enough that we return before the next frame arrives.
            Essentially this method just calls through to another appropriate method, based on the current value of the state attribute."""
        
        # For logging 
        time_init = time.perf_counter()
        self.currentTimeStamp = pixelArray.metadata["timestamp"]
        self.slow_action_occurred = None # Will be set to a descriptive string later, if applicable
        self.frame_num += 1
        self.frame_num_total += 1
        
        logger.info(
            "\n \n ################################################ Frame = {0} | Timestamp =  {1} s | State = {2} ################################################",
            self.frame_num_total,
            pixelArray.metadata["timestamp"],
            self.state
        )
        # Specify the reference sequence update criterion and initialise update_criterion accordingly
        if (self.settings["reference"]["ref_update_criterion"] == "frames"):
            self.update_criterion = self.frame_num
        elif (self.settings["reference"]["ref_update_criterion"] == "triggers"):
            self.update_criterion = self.trigger_num
        # Save first n frames
        if (not self.settings["brightfield"]["save_first_n_frames"] == None
            and len(self.frames_to_save) < self.settings["brightfield"]["save_first_n_frames"] 
           ):
            self.frames_to_save.append(pixelArray)
            if len(self.frames_to_save) == self.settings["brightfield"]["save_first_n_frames"]:
                ref.save_period(self.frames_to_save, self.settings["reference"]["reference_sequence_dir"], prefix="VID-")
        
        # Initiate a timelapse reference sequence update if a certain number of triggers 
        # has been reached in a given timelapse
        if self.timelapse_trigger_num >= self.settings["reference"]["triggers_between_timelapse"]:
            self.frame_num = 0
            self.trigger_num = 0
            self.timelapse_trigger_num = 0
            self.last_timelapse_time = self.currentTimeStamp
            self.state = "timelapse_pause"
          
        # Initiate a "normal" reference sequence update if a certain number of triggers
        # have been sent since the last update    
        elif (
            ("update_after_n_criterions" in self.settings["reference"]) and
            (self.update_criterion >= self.settings["reference"]["update_after_n_criterions"])
           ):
            # It is time to update the reference period (whilst maintaining phase lock)
            # As part of this reset, trigger_num and frame_num will both be reset
            logger.debug(
                        "Refreshing reference sequence (counter reached {0})",
                        self.settings["reference"]["update_after_n_criterions"]
                        )
            self.frame_num = 0
            self.trigger_num = 0
            self.state = "prep_basic_refresh"
        
        pixelArray.metadata["optical_gating_state"] = self.state
        if self.state == "prep_full_reset":
            self.prepare_for_state_change("full_reset")
        elif self.state == "full_reset":
            self.full_reset_state(pixelArray)
        elif self.state == "prep_basic_refresh":
            self.prepare_for_state_change("basic_refresh")
        elif self.state == "basic_refresh":
            self.basic_refresh_state(pixelArray)
        elif self.state == "timelapse_pause":
            self.timelapse_pause_state()
        elif self.state == "prep_timelapse_refresh":
            self.prepare_for_state_change("timelapse_refresh")
        elif self.state == "timelapse_refresh":
            self.timelapse_refresh_state(pixelArray)
        elif self.state == "sync":
            self.sync_state(pixelArray)
        else:
            logger.critical("Unknown state '{0}'", self.state)
            raise NotImplementedError("Unknown state '{0}'".format(self.state))
        
        # Logging crucial information
        if self.state == 'sync':
            if len(self.frame_history) > 0:
                logger.info(
                    "LOG TYPE A: Timestamp = {0} | State = {1} | Phase = {2} | Target Phase = {3}", 
                    pixelArray.metadata["timestamp"],
                    self.state,
                    self.frame_history[-1].metadata["unwrapped_phase"],
                    self.frame_history[-1].metadata["targetSyncPhase"]
                )
        else:
            logger.info(
                "LOG TYPE A: Timestamp = {0} | State = {1} | Phase = {2} | Target Phase = {3}", 
                pixelArray.metadata["timestamp"],
                self.state,
                None,
                None
            )
            
        # take a note of our processing rate (useful for deciding what camera framerate is viable to use)
        time_fin = time.perf_counter()
        pixelArray.metadata["processing_rate_fps"] = 1 / (
                time_fin - time_init
            )
            
    def prepare_for_state_change(self, target_state):
        """
        Called to prepare for a change of state from sync to a reset/refresh state.
        For a refresh state, the ref_seq_manager and predictor are re-initialised.
        For a full reset, 2 aligners are initialised - one for in-stack reference sequences,
        and one for trans-stack z = 0 reference sequences.
        """
        logger.debug("Preparing for state change")
        # Re-initialise the ref_seq_manager
        self.ref_seq_manager = ref.ReferenceSequenceManager(self.settings["reference"])
        # Initialise our predictor
        if self.settings["prediction"]["prediction_method"] == "kalman":
            # Kalman filter-based predictor
            logger.info("Initialising Kalman predictor")
            # TODO: Currently these are set to 'good enough' starting values - set to better estimates
            # TODO: initialise KF in predictor - don't need to define these here
            dt = 1 / self.settings["brightfield"]["brightfield_framerate"]#Setting to framerate of forced framerate (1/63) seems to fix this
            x0 = np.array([0, 10])
            P0 = np.array([[100, 100], [100, 100]])
            q = 1
            R = 0.1
            self.predictor = pog.KalmanPredictor(self.settings["prediction"]["kalman"], dt, x0, P0, q, R)
        elif self.settings["prediction"]["prediction_method"] == "linear":
            # Linear predictor
            logger.info("Initialising linear predictor")
            self.predictor = pog.LinearPredictor(self.settings["prediction"]["linear"])
        elif self.settings["prediction"]["prediction_method"] == "IMM":
            # Interacting multiple model Kalman filter predictor
            logger.info("Initialising IMM predictor")
            dt = 1 / self.settings["brightfield"]["brightfield_framerate"]#Setting to framerate of forced framerate (1/63) seems to fix this
            x0 = np.array([0, 10])
            P0 = np.array([[1000, 1000], [1000, 1000]])
            q = 1
            R = 0.00001
            self.predictor = pog.IMMPredictor(self.settings["prediction"]["IMM"], dt, x0, P0, q, R)
        else:
            logger.critical("Unknown prediction method '{0}'", self.settings["prediction"]["prediction_method"])
            raise NotImplementedError("Unknown prediction method '{0}'".format(self.settings["prediction"]["prediction_method"]))
        # Initialise 2 aligners in the case that a full_reset is desired
        if target_state == "full_reset":
            self.aligner1 = oga.Aligner(self.settings["oga"])
            self.aligner2 = oga.Aligner(self.settings["oga"])
        # Set state to target state to be moved onto in the next pass of analyze_pixelarray
        self.state = target_state
    
    def save_ref_frames(self, specialPrefix = "REF-"):
        # Save the most recent reference sequence
        if "reference_sequence_dir" in self.settings["reference"]:
            logger.debug("Reference sequence has been saved to disk")
            self.ref_seq_manager.save_ref_sequence(
                self.settings["reference"]["reference_sequence_dir"], 
                runIdentifier = "_" + str(self.settings["general"]["log_counter"]),
                prefix = specialPrefix
                )
        
    def full_reset_state(self, pixelArray):
        """
        Called to fully reset the synchronisation and establish an entirely new reference sequence.
        This is currently only called at the very start of the synchronisation, but
        could reasonably be called at other times.
        
        A reference sequence is generated -> target frame selected automatically or by the user
        -> both aligners updated with reference information.
        """
        logger.debug("Full reset initiated")
        if not hasattr(self, "supplied_ref_frames"):
            logger.debug("User has not supplied reference frames. Generating some...")
            ref_frames, period_to_use = self.ref_seq_manager.establish_period_from_frames(pixelArray)
        else:
            logger.debug("User has supplied reference frames")
            assert hasattr(self, "supplied_ref_frame_period")
            ref_frames, period_to_use = self.supplied_ref_frames, self.supplied_ref_frame_period
            
        if ref_frames is not None:
            # Set reference frames
            self.ref_seq_manager.set_ref_frames(ref_frames, period_to_use)
            self.save_ref_frames()
            self.slow_action_occurred = "reference frame refresh"
            # Select a target frame automatically or by input from the user
            # as implemented by subclasses
            method = self.settings["reference"]["target_frame_selection_method"]
            if method == "user":
                newTargetFrame, newBarrierFrame = self.user_pick_target_frame()
                logger.debug(f"User has selected target frame {newTargetFrame}")
                ok = (newTargetFrame >= 0)
            elif method == "auto":
                newTargetFrame = None
                newBarrierFrame = None
            self.set_target_frame(newTargetFrame, newBarrierFrame)
            
            # Add reference information to both the aligners as this first sequence will 
            # correspond to the z = 0 plane
            self.aligner1.process_initial_sequence(
                self.ref_seq_manager.ref_frames,
                self.ref_seq_manager.ref_period,
                self.ref_seq_manager.drift,
                self.ref_seq_manager.targetFrameNum
                )
            self.aligner2.process_initial_sequence(
                self.ref_seq_manager.ref_frames,
                self.ref_seq_manager.ref_period,
                self.ref_seq_manager.drift,
                self.ref_seq_manager.targetFrameNum
                )
            self.state = "sync"
                
    def basic_refresh_state(self, pixelArray):
        """
        Called when a reference sequence refresh is required (i.e when the brightfield has
        moved sufficiently out of plane of the previous reference sequence).
        
        New reference sequence is establised and saved-> aligner selects appropriate target frame.
        """
        logger.debug("Performing a basic reference refresh")
        ref_frames, period_to_use = self.ref_seq_manager.establish_period_from_frames(pixelArray)
        if ref_frames is not None:
            logger.debug("Established reference sequence with period {0}".format(period_to_use))
            self.ref_seq_manager.set_ref_frames(ref_frames, period_to_use)
            self.save_ref_frames()
            self.slow_action_occurred = "reference frame adaptive update"

            # Find and set the new target frame
            newTargetFrame = self.aligner1.process_sequence(
                self.ref_seq_manager.ref_frames,
                self.ref_seq_manager.ref_period,
                self.ref_seq_manager.drift
                )
            self.set_target_frame(newTargetFrame, None)
            self.state = "sync"
    
    def timelapse_pause_state(self):
        logger.debug("Pausing for timelapse...")
        if self.currentTimeStamp - self.last_timelapse_time >= self.settings["general"]["pause_for_timelapse"]:
            self.state = "prep_timelapse_refresh"
            
    def timelapse_refresh_state(self, pixelArray):
        """
        Called when the user returns to the z = 0 plane at the end of a stack acquisition.
        New incoming brightfield frames may appear very different to a reference sequence 
        acquired at, for example, z = 200. We must cross-correlate with the original z = 0
        reference sequence to ensure an accurate phase lock is achieved.
        
        A new reference sequence is acquired -> aligner 2 selects the appropriate target frame
        -> aligner 1 is re-initiased to discard the reference sequences from the previous stack
        -> the new aligner 1 is updated with the new reference sequence and target frame.
        """
        
        logger.debug("Performing a timelapse refresh")
        ref_frames, period_to_use = self.ref_seq_manager.establish_period_from_frames(pixelArray)
        if ref_frames is not None:
            self.ref_seq_manager.set_ref_frames(ref_frames, period_to_use)
            self.save_ref_frames(specialPrefix = "REF-timelapse-")
            # Find and set the appropriate target frame for the new z = 0 reference sequence
            newTargetFrame= self.aligner2.process_sequence(
                self.ref_seq_manager.ref_frames, 
                self.ref_seq_manager.ref_period, 
                self.ref_seq_manager.drift
                )
            self.set_target_frame(newTargetFrame, None)
            self.slow_action_occurred = "reference frame adaptive update"
            logger.debug(f"A new target frame ({newTargetFrame}) has been generated")
            
            # Re-initialising aligner 1 and add new reference sequence and target frame
            self.aligner1 = oga.Aligner(self.settings["oga"])
            self.aligner1.process_initial_sequence(
                self.ref_seq_manager.ref_frames,
                self.ref_seq_manager.ref_period, 
                self.ref_seq_manager.drift,
                self.ref_seq_manager.targetFrameNum
            )
            # Return to sync state
            self.state = "sync"
            
    def set_target_frame(self, new_target_frame, new_barrier):
        """ Impose a new target frame representing the heart phase that our code will aim to synchronize to.
            If input values are None, we will make our own empirical guess of values we think should perform well
        """
        print(f"New target frame = {new_target_frame}")
        logger.debug("set_target_frame() called with {0}, {1}", new_target_frame, new_barrier)
        defaultTarget, defaultBarrier = self.ref_seq_manager.pick_good_target_and_barrier_frames()
        if new_target_frame is None:
            new_target_frame = defaultTarget
        if new_barrier is None:
            new_barrier = defaultBarrier
        logger.debug("Setting reference and barrier frames to {0}, {1}", new_target_frame, new_barrier)
        self.ref_seq_manager.set_target_and_barrier_frame(new_target_frame, new_barrier)
        self.predictor.target_and_barrier_updated(self.ref_seq_manager)

    def sync_state(self, pixelArray):
        """ Code to run when in "sync" state
            Perform prospective optical gating for heartbeat-synchronized triggering.
        """
        logger.debug("Processing frame in prospective optical gating mode.")

        # Gets the phase (in frames) and arrays of SADs between the current frame and the reference sequence
        current_phase, sad = self.ref_seq_manager.identify_phase_for_frame(pixelArray)
        logger.debug("SAD curve: {0}", sad)

        # Calculate the unwrapped phase.
        if len(self.frame_history) == 0:
            # This is our first frame - the unwrapped phase is simply the phase of this frame
            delta_phase = 0
            phase = current_phase
        else:
            # Normally it is fairly straightforward to determine the unwrapped phase:
            # we just calculate (current_phase - previous_phase) and add that to the unwrapped phase of the previous frame.
            # But some care is needed to handle the case where the current phase has just wrapped,
            # or there is a slight backward step from e.g. 0.01 to 2π-0.01.
            previous_phase = self.frame_history[-1].metadata["unwrapped_phase"] % (2*np.pi)
            delta_phase = current_phase - previous_phase
            # Handle phase wraps in the most sensible way possible
            # (when mapped to [0,2π], we consider [0,π] to be a forward step and [π,2π] to be backwards.
            while delta_phase < -np.pi:
                delta_phase += 2 * np.pi
            phase = self.frame_history[-1].metadata["unwrapped_phase"] + delta_phase

        # Limit the length of frame_history by evicting the oldest entry
        # if the list length exceeds the maximum length we are meant to be retaining.
        # Note: deletion of the *first* element of a list is potentially a performance issue,
        # although we are hopefully capping the length low enough that it doesn't become a real bottleneck
        if len(self.frame_history) >= self.settings["general"]["frame_buffer_length"]:
            del self.frame_history[0]

        # Append our current PixelArray object (including its metadata) to our frame_history list
        thisFrameMetadata = pixelArray.metadata
        thisFrameMetadata["unwrapped_phase"] = phase
        thisFrameMetadata["sad_min"] = np.argmin(sad)
        thisFrameMetadata["phase"] = current_phase
        thisFrameMetadata["delta_phase"] = delta_phase
        self.frame_history.append(pixelArray)

        logger.debug(
            "Current time: {0} s; cumulative phase: {1} (delta:{2:+f}) rad; sad min: {3}",
            thisFrameMetadata["timestamp"],
            thisFrameMetadata["unwrapped_phase"],
            delta_phase,
            thisFrameMetadata["sad_min"],
        )

        # === The main purpose of this function: generating synchronization triggers ===
        # If we have at least one period of phase history, have a go at predicting a future trigger time
        # (Note that this prediction can be disabled by including a "phase_stamp_only" key in the settings file
        frameInterval_s = 1.0 / self.settings["brightfield"]["brightfield_framerate"]
        this_predicted_trigger_time_s = None
        sendTriggerReason = None
        if ((len(self.frame_history) > self.ref_seq_manager.ref_period)
            and (not "phase_stamp_only" in self.settings)
        ):
            logger.debug("Predicting trigger...")

            # Make a future prediction
            logger.trace("Predicting next trigger.")
            timeToWait_s, estHeartPeriod_s = self.predictor.predict_trigger_wait(
                self.frame_history,
                self.ref_seq_manager.targetSyncPhase,
                frameInterval_s,
                fitBackToBarrier=True
            )
            logger.trace("Time to wait to trigger: {0} s.".format(timeToWait_s))

            this_predicted_trigger_time_s = thisFrameMetadata["timestamp"] + timeToWait_s

            # If we have a prediction, consider actually sending the trigger
            if timeToWait_s > 0:
                logger.debug("Possible trigger after: {0}s", timeToWait_s)
                # Decide whether the current candidate trigger time should actually be used,
                # or whether we should wait for an improved prediction from the next brightfield frame.
                # Note that timeToWait_s might be updated by this call (to change the value to
                # refer to the next heart cycle) if we have already committed to a trigger in the current cycle.
                #
                # JT TODO: Check if I want to do more refactoring here (and/or match better with Ross's Kalman code)
                (
                    timeToWait_s,
                    sendTriggerReason
                ) = self.predictor.decide_whether_to_trigger(
                    thisFrameMetadata["timestamp"],
                    timeToWait_s,
                    frameInterval_s,
                    estHeartPeriod_s
                )
                
                if (
                    sendTriggerReason is not None
                    and this_predicted_trigger_time_s - self.mostRecentTriggerTime >= self.settings["general"]["min_time_between_triggers"]
                    ):
                    self.mostRecentTriggerTime = this_predicted_trigger_time_s
                    # Actually send the electrical trigger signal
                    logger.debug(
                        "Sending trigger (reason: {0}) at time ({1} + {2}) s",
                        sendTriggerReason,
                        thisFrameMetadata["timestamp"],
                        timeToWait_s,
                    )
                    
                    # Note that the following call may block on some platforms
                    # (its exact implementation is for the subclass to determine)
                    timeNow = thisFrameMetadata["timestamp"]
                    self.trigger_fluorescence_image_capture(
                        this_predicted_trigger_time_s
                    )
                    # Track the frame that initiated the trigger, because we will go back later
                    # and update its metadata with metrics of how good we think the trigger was
                    self.latestTriggerPredictFrame = self.frame_history[-1]
                    # trigger_sent is a special flag that is *only* present in the dictionary if we did send a trigger
                    thisFrameMetadata["trigger_sent"] = (sendTriggerReason is not None)

                    # Update trigger iterator (for adaptive algorithm)
                    self.trigger_num += 1
                    self.timelapse_trigger_num += 1
                    self.trigger_num_total +=1
                    
                    logger.info(
                        'LOG TYPE B: Trigger Decided at = {0} seconds; sent at {1} seconds'.format(timeNow, this_predicted_trigger_time_s)
                        )

        # Update PixelArray with predicted trigger time and trigger type
        thisFrameMetadata["predicted_trigger_time_s"] = this_predicted_trigger_time_s
        thisFrameMetadata["trigger_type_sent"] = sendTriggerReason
        thisFrameMetadata["targetSyncPhase"] = self.ref_seq_manager.targetSyncPhase

        logger.debug(
            "Sync analysis completed. Current time: {0} s; predicted trigger time: {1} s; trigger type: {2}; brightfield frame number: {3}",
            thisFrameMetadata["timestamp"],
            thisFrameMetadata["predicted_trigger_time_s"],
            thisFrameMetadata["trigger_type_sent"],
            self.frame_num_total,
        )

        # Retrospective monitoring of how closely a recent previous trigger ended up matching the target phase
        self.live_phase_interpolation()

    def trigger_fluorescence_image_capture(self, trigger_time_s):
        """
        As this is the base server, this function just outputs a log that a trigger would have been sent.
        Subclasses should override this if they want to interact with hardware.
        """
        logger.debug("A fluorescence image would be triggered now.")

    def live_phase_interpolation(self): 
        """
        Fluorescence triggers will rarely (if ever) overlap exactly in time with brightfield frames. 
        To achieve an accurate phase estimate at the time a fluorescence image was captured,
        it is necessary to interpolated between times of known phase.
        
        This function interpolates phase between the TWO CLOSEST brightfield frames
        to a given sent trigger time in order to the estimate phase
        at the exact time a fluorescence frame was captured. 
        """
        
        # If a trigger was previously scheduled and will have fired just before the frame we are now processing,
        # perform phase interpolation to estimate what the heart phase actually was when the trigger fired
        if (self.latestTriggerPredictFrame is not None     # Ensures there is at least one sent trigger to analyse
            and len(self.frame_history) > 1):              # Ensures there are two frames to interpolate between
            
            aheadTime = self.frame_history[-1].metadata["timestamp"]
            behindTime = self.frame_history[-2].metadata["timestamp"]
            triggerTime = self.latestTriggerPredictFrame.metadata["predicted_trigger_time_s"]
            if (aheadTime >= triggerTime        # Ensure the latest brightfield image is 'ahead' of the recently-sent trigger time
                and behindTime <= triggerTime): # Ensure the previous brightfield image is 'behind' the recently-sent trigger time
            
                aheadPhase = self.frame_history[-1].metadata["unwrapped_phase"]
                behindPhase = self.frame_history[-2].metadata["unwrapped_phase"]

                interpolatedPhase = np.interp(triggerTime, [behindTime, aheadTime], [behindPhase, aheadPhase])
                wrappedPhaseAtSentTriggerTime = interpolatedPhase % (2 * np.pi)
            
                # Compute the error between the target phase and the estimated phase
                phaseError = wrappedPhaseAtSentTriggerTime - self.ref_seq_manager.targetSyncPhase
            
                # Adjust the phase error to lie in (-pi, pi)
                if phaseError > np.pi:
                    phaseError = phaseError - (2 * np.pi)
                elif phaseError < - np.pi:
                    phaseError = phaseError + (2 * np.pi)
            
                self.latestTriggerPredictFrame.metadata["triggerPhaseError"] = phaseError
                self.latestTriggerPredictFrame.metadata["wrappedPhaseAtSentTriggerTime"] = wrappedPhaseAtSentTriggerTime

                logger.debug('Live phase interpolation successful! Phase error = {0}', phaseError)

    def plot_triggers(self, outfile="triggers.png"):
        """
        Plot the phase vs. time sawtooth line with trigger events.
        """
        # get trigger times from predicted triggers time and trigger types sent (e.g. not 0)
        sent_trigger_times = pa.get_metadata_from_list(
            self.frame_history,
            "predicted_trigger_time_s",
            onlyIfKeyPresent="trigger_sent"
        )
        sent_trigger_target_phases = pa.get_metadata_from_list(
            self.frame_history,
            "targetSyncPhase",
            onlyIfKeyPresent="trigger_sent"
        )
        plt.figure()
        plt.title("Zebrafish heart phase with trigger fires")
        plt.plot(
            pa.get_metadata_from_list(self.frame_history, "timestamp"),
            pa.get_metadata_from_list(self.frame_history, "unwrapped_phase")
            % (2 * np.pi),
            color = "tab:green",
            label="Heart phase",
            zorder=5
        )
        plt.scatter(
            np.array(sent_trigger_times),
            np.array(sent_trigger_target_phases),
            color="tab:red",
            label="Trigger fire",
            zorder = 10
        )
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (rad)")
        plt.show()
        
    def plot_phase_histogram(self):
        """
        Plots a histogram representing the frequency density of triggered phases (green) in addition to markers denoting the target phase (red).
        """
        plt.figure()
        plt.title("Frequency density of triggered phase")
        plt.hist(
            pa.get_metadata_from_list(self.frame_history, "wrappedPhaseAtSentTriggerTime", onlyIfKeyPresent="wrappedPhaseAtSentTriggerTime"),
            bins = np.arange(0, 2 * np.pi, 0.01), 
            color = "tab:green", 
            label = "Triggered phase"
        )
        x_1, x_2, y_1, y_2 = plt.axis()

        # Add red markers indicating the target phases
        uniqueTargetPhases = np.unique(pa.get_metadata_from_list(self.frame_history, "targetSyncPhase"))
        for ph in uniqueTargetPhases:
            plt.plot(np.full(2, ph), (y_1, y_2),"red", label = "Target phase",)
        plt.xlabel("Triggered phase (rad)")
        plt.ylabel("Frequency")
        plt.axis((x_1, x_2, y_1, y_2))
        plt.tight_layout()
        plt.show()
        
    def plot_phase_error_histogram(self):
        """
        Plots a histogram representing the frequency estimated phase errors.
        """
        phaseErrorList = pa.get_metadata_from_list(
            self.frame_history, 
            "triggerPhaseError", 
            onlyIfKeyPresent="triggerPhaseError"
        )
        plt.figure()
        plt.title("Histogram of triggered phase errors")
        plt.hist(
            phaseErrorList,
            bins = np.arange(np.min(phaseErrorList), np.max(phaseErrorList) + 0.1,  0.03), 
            color = "tab:green", 
            label = "Triggered phase"
        )
        x_1, x_2, y_1, y_2 = plt.axis()
        plt.xlabel("Frequency density of phase error (rad)")
        plt.ylabel("Frequency")
        plt.axis((x_1, x_2, y_1, y_2))
        plt.tight_layout()
        plt.show()
    
    def plot_phase_error_with_time(self):
        """
        Plots the estimated phase error associated with each sent trigger over time.
        """
        phaseErrorList = pa.get_metadata_from_list(
            self.frame_history, 
            "triggerPhaseError", 
            onlyIfKeyPresent="triggerPhaseError"
        )
        timeList = pa.get_metadata_from_list(
                self.frame_history, 
                "predicted_trigger_time_s", 
                onlyIfKeyPresent="trigger_sent"
        )
        plt.figure()
        plt.title('Triggered phase error with time')
        plt.scatter(
            timeList[:len(phaseErrorList)], 
            phaseErrorList,
            color = 'tab:green'
        )
        plt.xlabel('Time (s)')
        plt.ylabel('Phase error (rad)')
        plt.ylim(-np.pi, np.pi)
        plt.tight_layout()
        plt.show()

    def plot_prediction(self):
        plt.figure()
        plt.title("Predicted trigger times")
        plt.plot(
            pa.get_metadata_from_list(self.frame_history, "timestamp"),
            pa.get_metadata_from_list(self.frame_history, "predicted_trigger_time_s"),
            color = "tab:green"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Prediction (s)")
        plt.show()

    def plot_running(self):
        plt.figure()
        plt.title("Frame processing rate")
        plt.plot(
            pa.get_metadata_from_list(self.frame_history, "timestamp"),
            pa.get_metadata_from_list(self.frame_history, "processing_rate_fps"),
            color = "tab:green"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Processing rate (fps)")
        plt.show()

    def plot_delta_phase_phase(self):
        plt.figure()
        plt.scatter(pa.get_metadata_from_list(self.frame_history, "phase"), pa.get_metadata_from_list(self.frame_history, "delta_phase"))
        plt.show()

    def plot_likelihood(self):
        plt.figure()
        plt.scatter(pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent="likelihood"), pa.get_metadata_from_list(self.frame_history, "likelihood", onlyIfKeyPresent="likelihood"))
        plt.show()