"""Parent Open Optical Gating Class"""

# Python imports
import sys, time
import json

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

# Set up logging
logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("testing_logs/oog_{time}.log", level="DEBUG")
logger.enable("open_optical_gating")

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
logger.info("Running in module environment: {0}", version_dict)

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

    def __init__(self, settings=None, ref_frames=None, ref_frame_period=None):
        """Function inputs:
            settings - a dictionary of settings (see optical_gating_data/json_format_description.md)
        """

        # store the whole settings dict
        # we occasionally store some of this information elsewhere too
        # that's not ideal but works for now
        self.settings = settings
        
        logger.info("Instantiated OpticalGater with settings: {0}", settings)

        logger.success("Initialising internal parameters...")
        self.initialise_internal_parameters()
        self.reset_state()
        if ref_frames is not None:
            self.ref_seq_manager.set_ref_frames(ref_frames, ref_frame_period)
            self.state = "sync"
            # We need to identify the target frame.
            # If the caller cares what it is, they should specify it via the 'settings' dictionary
            ok = self.identify_and_set_reference_frame()
            assert(ok)
    
    def initialise_internal_parameters(self):
        """Defines all internal parameters not already initialised"""
        self.frame_history = []
        self.frames_to_save = []
        self.slow_action_occurred = None
        self.latestTriggerPredictFrame = None
        
        self.trigger_num = 0
        self.frame_num = 0
        self.frame_num_total = 0
        self.aligner = oga.Aligner(self.settings["oga"])

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
            (self.aligner.sequence_history is None)
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
        logger.debug("Analysing frame with timestamp: {0}s", pixelArray.metadata["timestamp"])
        self.slow_action_occurred = None # Will be set to a descriptive string later, if applicable
        self.frame_num += 1 
        self.frame_num_total += 1
        # Specify the reference sequence update criterion and initialise update_criterion accordingly
        if (self.settings["ref_update_criterion"] == "frames"):
            self.update_criterion = self.frame_num
        elif (self.settings["ref_update_criterion"] == "triggers"):
            self.update_criterion = self.trigger_num
            
        # For logging processing time
        time_init = time.perf_counter()
        
        if (
            ("update_after_n_criterions" in self.settings) and
            (self.update_criterion >= self.settings["update_after_n_criterions"])
           ):
            # It is time to update the reference period (whilst maintaining phase lock)
            # Set state to "reset" (so we clear things for a new reference period)
            # As part of this reset, trigger_num and frame_num will both be reset
            logger.info(
                        "Refreshing reference sequence (counter reached {0})",
                        self.settings["update_after_n_criterions"]
                        )
            self.frame_num = 0
            self.trigger_num = 0
            self.state = "reset"

        pixelArray.metadata["optical_gating_state"] = self.state
        
        if (
            ("save_first_n_frames" in self.settings) and
            (len(self.frames_to_save) < self.settings["save_first_n_frames"])
           ):
            self.frames_to_save.append(pixelArray)
            if len(self.frames_to_save) == self.settings["save_first_n_frames"]:
                ref.save_period(self.frames_to_save, self.settings["reference_sequence_dir"], prefix="VID-")
            
        if self.state == "reset":
            # Clears reference period and resets frame number
            # Used when determining new period
            self.reset_state()
        elif self.state == "determine":
            # Determine initial reference period and target frame
            self.determine_state(pixelArray)
        elif self.state == "sync":
            # Using previously-determined reference period, analyse brightfield frames
            # to determine predicted trigger time for prospective optical gating
            self.sync_state(pixelArray)
        else:
            logger.critical("Unknown state '{0}'", self.state)
            raise NotImplementedError("Unknown state '{0}'".format(self.state))

        # take a note of our processing rate (useful for deciding what camera framerate is viable to use)
        time_fin = time.perf_counter()
        pixelArray.metadata["processing_rate_fps"] = 1 / (
                time_fin - time_init
            )

    def reset_state(self):
        """ Code to run when resetting all state (ready to determine a new period)
            Used if the user is not happy with a period choice,
            or before getting a new reference period in the adaptive mode.
        """
        logger.info("Resetting for new period determination.")
        self.ref_seq_manager = ref.ReferenceSequenceManager(self.settings["reference_finding"])
        self.predictor = pog.LinearPredictor(self.settings["linear_prediction"])
        self.state = "determine"
    
    def determine_state(self, pixelArray):
        """ Code to run when in "determine" state (identifying a reference sequence).
            In this mode we accumulate frames and try and recognise a single heartbeat sequence from within them.
        """
        logger.debug("Processing frame to determine period")

        ref_frames, period_to_use = self.ref_seq_manager.establish_period_from_frames(pixelArray)
        if ref_frames is not None:
            logger.success("Established reference sequence with period {0}".format(period_to_use))
            self.ref_seq_manager.set_ref_frames(ref_frames, period_to_use)
            if "reference_sequence_dir" in self.settings:
                # Save the reference sequence to disk, for debug purposes
                self.ref_seq_manager.save_ref_sequence(self.settings["reference_sequence_dir"])
            self.slow_action_occurred = "reference frame refresh"

            if self.identify_and_set_reference_frame():
                self.state = "sync"
                logger.success(
                           "Period determined ({0} frames long) and we have selected frame {1} as target.",
                           self.ref_seq_manager.ref_period,
                           self.ref_seq_manager.targetFrameNum
                           )
            else:
                # Latest reference sequence was rejected for some reason.
                # Start from scratch
                logger.debug("Reference sequence rejected - resetting")
                self.state = "reset"

    def sync_state(self, pixelArray):
        """ Code to run when in "sync" state
            Perform prospective optical gating for heartbeat-synchronized triggering.
        """
        logger.debug("Processing frame in prospective optical gating mode.")

        # Gets the phase (in frames) and arrays of SADs between the current frame and the reference sequence
        current_phase, sad = self.ref_seq_manager.identify_phase_for_frame(pixelArray)
        logger.debug("SAD: {0}", sad)

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
        if len(self.frame_history) >= self.settings["frame_buffer_length"]:
            del self.frame_history[0]

        # Append our current PixelArray object (including its metadata) to our frame_history list
        thisFrameMetadata = pixelArray.metadata
        thisFrameMetadata["unwrapped_phase"] = phase
        thisFrameMetadata["sad_min"] = np.argmin(sad)
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
        frameInterval_s = 1.0 / self.settings["brightfield_framerate"]
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
                logger.info("Possible trigger after: {0}s", timeToWait_s)
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
                if sendTriggerReason is not None:
                    # Actually send the electrical trigger signal
                    logger.success(
                        "Sending trigger (reason: {0}) at time ({1} + {2}) s",
                        sendTriggerReason,
                        thisFrameMetadata["timestamp"],
                        timeToWait_s,
                    )
                    # Note that the following call may block on some platforms
                    # (its exact implementation is for the subclass to determine)
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

                    logger.debug(
                        'Retrospective Log Analysis Data (Type B): Trigger Time = {0}'.format(this_predicted_trigger_time_s)
                        )

        # Update PixelArray with predicted trigger time and trigger type
        thisFrameMetadata["predicted_trigger_time_s"] = this_predicted_trigger_time_s
        thisFrameMetadata["trigger_type_sent"] = sendTriggerReason
        thisFrameMetadata["targetSyncPhase"] = self.ref_seq_manager.targetSyncPhase

        logger.debug(
            "Sync analysis completed. Current time: {0} s; predicted trigger time: {1} s; trigger type: {2}; brightfield frame {3}",
            thisFrameMetadata["timestamp"],
            thisFrameMetadata["predicted_trigger_time_s"],
            thisFrameMetadata["trigger_type_sent"],
            self.frame_num_total,
        )

        # Retrospective monitoring of how closely a recent previous trigger ended up matching the target phase
        self.live_phase_interpolation()

        logger.debug(
            'Retrospective Log Analysis Data (Type A): Timestamp = {0} Phase = {1} Target Phase = {2}'.format(
                thisFrameMetadata["timestamp"],
                thisFrameMetadata["unwrapped_phase"],
                thisFrameMetadata["targetSyncPhase"],
            )
        )

    def identify_and_set_reference_frame(self):
        """ Select a reference frame. We may:
            - Use a value hard-coded in the config settings
            - Prompt the user to pick one
            - Make a guess of a suitable one to use
            - Use the LTU code to maintain the same target phase as was previously in force
            
            Returns True if a suitable reference frame has been applied,
             or False if we need to start the period-determining process from scratch again.
        """
        ok = True
        method = self.settings["reference_finding"]["target_frame_selection_method"]
        adaptive = self.settings["reference_finding"]["target_frame_adaptive_update"]

        if (adaptive and (self.aligner.sequence_history is not None)):
            # Automatically maintain existing target phase
            logger.info("Use LTU code to compute the reference frame")
            newTargetFrame, newBarrierFrame = self.pick_target_frame_adaptively()
        elif method == "config":
            # Config file specifies reference frame
            logger.info("Config file specifies reference frame of {0}", self.settings["reference_finding"]["target_frame_default"])
            newTargetFrame = self.settings["reference_finding"]["target_frame_default"]
            newBarrierFrame = None
        elif method == "user":
            # User types a choice
            logger.info("User will select a reference frame")
            newTargetFrame, newBarrierFrame = self.user_pick_target_frame()
            ok = (newTargetFrame >= 0)
        elif method == "auto":
            # Automatically pick a frame
            logger.info("Automatically pick a suitable reference frame")
            newTargetFrame = None
            newBarrierFrame = None
                
        if ok:
            # We decided on a reference frame to use
            self.set_target_frame(newTargetFrame, newBarrierFrame)
            if (adaptive and (self.aligner.sequence_history is None)):
                # We need to seed the LTU code with the first reference sequence information.
                # JT TODO: somewhere we need to support the case where the user adjusts the reference frame in the middle of a run.
                # In that situation we need to decide whether to use ref_seq_id!=0, or to clear all the OGA history and start afresh
                #            self.pog_settings["oga_reference_value"] = self.pog_settings["referenceFrame"]
                
                # JT TODO: it would be nice if there was a way to avoid having to do this separately to the call to oga.process_sequence.
                self.aligner.process_initial_sequence(self.ref_seq_manager.ref_frames,
                                                      self.ref_seq_manager.ref_period,
                                                      self.ref_seq_manager.drift,
                                                      self.ref_seq_manager.targetFrameNum)

        return ok

    def set_target_frame(self, new_target_frame, new_barrier):
        """ Impose a new target frame representing the heart phase that our code will aim to synchronize to.
            If input values are None, we will make our own empirical guess of values we think should perform well
        """
        logger.debug("set_target_frame() called with {0}, {1}", new_target_frame, new_barrier)
        defaultTarget, defaultBarrier = self.ref_seq_manager.pick_good_target_and_barrier_frames()
        if new_target_frame is None:
            new_target_frame = defaultTarget
        if new_barrier is None:
            new_barrier = defaultBarrier
        logger.success("Setting reference and barrier frames to {0}, {1}", new_target_frame, new_barrier)
        self.ref_seq_manager.set_target_and_barrier_frame(new_target_frame, new_barrier)
        # JT TODO: this function is called when we acquire a new set of reference frames,
        # but it is not updated if the parameters such as barrierFrame, min/maxFramesForFit
        # are later altered by the user. It should be...
        self.predictor.target_and_barrier_updated(self.ref_seq_manager)

    def pick_target_frame_adaptively(self):
        """ Adaptive prospective optical gating mode
            i.e. update reference sequence, while maintaining the same phase-lock.
            In this mode we align the latest reference sequence with
            previous sequences using an adaptive algorithm.
        """
        # Align the current reference sequence relative to previous ones (adaptive update)
        # Note that the ref_seq_phase parameter for process_sequence is in units of ??[WHAT?]??
        # JT TODO: update the above comment to clarify
        newTargetFrame = self.aligner.process_sequence(self.ref_seq_manager.ref_frames,
                                                    self.ref_seq_manager.ref_period,
                                                    self.ref_seq_manager.drift)
            
        logger.success("Reference frame adaptive update complete. New reference frame will be {0}", newTargetFrame)
        self.slow_action_occurred = "reference frame adaptive update"
        # JT TODO: currently this does not adaptively update the barrier frame,
        # it just lets the code identify the barrier frame from scratch without reference to anything specified previously
        return newTargetFrame, None

    def user_pick_target_frame(self):
        """Prompts the user to select the target frame from a one-period set of reference frames"""
        # For now it is a simple command line interface (which is not very user-friendly as you can't see the images)
        choice = input(
                        "Please select a target frame between 0 and "
                        + str(len(self.ref_seq_manager.ref_frames) - 1)
                        + "\nOr enter -1 to select a new period.\n"
                       )
        self.slow_action_occurred = "user selection of target frame"
        # JT TODO: currently we do not prompt the user about the barrier frame
        # We just let the code identify the barrier frame from scratch without reference to anything specified previously
        return int(choice), None

    def trigger_fluorescence_image_capture(self, trigger_time_s):
        """
        As this is the base server, this function just outputs a log that a trigger would have been sent.
        Subclasses should override this if they want to interact with hardware.
        """
        logger.success("A fluorescence image would be triggered now.")

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

                logger.info('Live phase interpolation successful! Phase error = {0}', phaseError)
            
                errorThreshold = 0.5
                if abs(phaseError) > errorThreshold:
                    logger.warning('Phase error ({0} radians) has exceeded desired threshold ({1} radians)', phaseError, errorThreshold)


    def plot_triggers(self, outfile="triggers.png"):
        """
        Plot the phase vs. time sawtooth line with trigger events.
        """
        # get trigger times from predicted triggers time and trigger types sent (e.g. not 0)
        sent_trigger_times = pa.get_metadata_from_list(self.frame_history,
                                                       "predicted_trigger_time_s",
                                                       onlyIfKeyPresent="trigger_sent")
        sent_trigger_target_phases = pa.get_metadata_from_list(self.frame_history,
                                                               "targetSyncPhase",
                                                               onlyIfKeyPresent="trigger_sent")

        plt.figure()
        plt.title("Zebrafish heart phase with trigger fires")
        plt.plot(
            pa.get_metadata_from_list(self.frame_history, "timestamp"),
            pa.get_metadata_from_list(self.frame_history, "unwrapped_phase")
            % (2 * np.pi),
            label="Heart phase",
        )
        plt.scatter(
            np.array(sent_trigger_times),
            np.array(sent_trigger_target_phases),
            color="r",
            label="Trigger fire",
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
        plt.figure()
        plt.title("Histogram of triggered phase errors")
        phaseErrorList = pa.get_metadata_from_list(
            self.frame_history, 
            "triggerPhaseError", 
            onlyIfKeyPresent="triggerPhaseError"
        )
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
        plt.figure()
        plt.title('Triggered phase error with time')
        plt.scatter(
            pa.get_metadata_from_list(self.frame_history, "predicted_trigger_time_s", onlyIfKeyPresent="trigger_sent"), 
            pa.get_metadata_from_list(self.frame_history, "triggerPhaseError", onlyIfKeyPresent="trigger_sent"), 
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
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Processing rate (fps)")
        plt.show()
