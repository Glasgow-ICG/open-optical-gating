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
from . import pog_settings as ps

logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("testing_logs/oog_{time}.log", level="DEBUG")
logger.enable("open_optical_gating")

# TODO create a time-stamped copy of the settings file after this
# TODO create a time-stamped log somewhere


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
        # NOTE: there is also self.pog_settings, be careful of this

        logger.success("Initialising internal parameters...")
        self.initialise_internal_parameters()
        
        if ref_frames is None:
            logger.info(
                        "No reference frames found, resetting before switching to determine period mode."
                        )
            self.state = "reset"
        else:
            logger.success("Using existing reference frames, setting to prospective optical gating mode.")
            if ref_frame_period is None:
                logger.warning("No period supplied, inferring integer period.")
                # Deduce an integer reference period from the reference frames we were provided with.
                # This is just a legacy mode - caller who constructed this object should really have provided a reference period
                ref_frame_period = ref_frames.shape[0] - 2 * pog.numExtraRefFrames
            self.pog_settings.set_reference_frames(ref_frames, ref_frame_period)
            self.state = "sync"

    def initialise_internal_parameters(self):
        """Defines all internal parameters not already initialised"""
        # Defines an empty list to store past frames with timestamp, phase and argmin(sad) metadata
        self.frame_history = []
        self.frames_to_save = []
        self.automatic_target_frame_selection = True
        self.justRefreshedRefFrames = False
        self.latestTriggerPredictFrame = None
        
        # Variables for adaptive algorithm
        self.trigger_num = 0
        self.sequence_history = None
        self.period_history = None
        self.drift_history = None
        self.shift_history = None
        self.global_solution = None

        # TODO: JT writes: this seems as good a place as any to flag the fact that I don't think barrier frames are being implemented properly.
        # There is a call to determine_barrier_frames, but I don't think the *value* for the barrier frame parameter is ever computed, is it?
        # It certainly isn't when using existing reference frames. This seems like an important missing bit of code.
        # I think it just defaults to 0 when the settings are initialised, and stays that way.

        self.pog_settings = ps.POGSettings({"framerate": self.settings["brightfield_framerate"]})

        # Start experiment timer
        self.initial_process_time_s = time.time()

        # Flag for interrupting the program at key points
        # E.g. when user-input is needed
        # It is assumed that the user/app controls what this interaction is
        self.stop = False

    def run_server(self):
        """ Run the OpticalGater server, acting on the supplied frame images.
            run_and_analyze_until_stopped() is implemented by subclasses, which call back into
            analyze_pixelarray() with the appropriate frame data to be analyzed
        """
        if self.automatic_target_frame_selection == False:
            logger.success("Determining reference period...")
            
            # Prompt for a target frame if reference frames are provided
            if self.pog_settings["ref_frames"] is not None:
                self.user_select_ref_frame()
            
            while self.state != "sync":
                # JT TODO: the current code does not quite work in the case where the user rejects the references by typing -1.
                # It sets state to "reset" but self.stop will be non-False *until* we go through the reset state action...
                # but we don't do anything here if self.stop is non-False!
                # I suspect the best fix would be a proper state machine that understands when we are acquiring a period,
                # but where some of the current "states" such as reset are actually state machine actions that take place
                # on certain state transitions.
                # For now this is a workaround that forces analyze_pixelarray to run the first time
                self.stop = False
                self.run_and_analyze_until_stopped()
                if self.stop == 'out-of-frames':
                    raise RuntimeError("Ran out of frames without managing to establish a period")
                logger.info("Requesting user input for ref frame selection...")
                self.user_select_ref_frame()
                
            logger.success(
                "Period determined ({0} frames long) and user has selected frame {1} as target.",
                self.pog_settings["reference_period"],
                self.pog_settings["referenceFrame"],
            )
                
        logger.success("Synchronizing...")
        self.run_and_analyze_until_stopped()
    
    def analyze_pixelarray(self, pixelArray):
        """ Method to analyse each frame as they are captured by the camera.
            Note that this analysis must take place fast enough that we return before the next frame arrives.
            Essentially this method just calls through to another appropriate method, based on the current value of the state attribute."""
        logger.debug(
            "Analysing frame with timestamp: {0}s", pixelArray.metadata["timestamp"],
        )
        self.justRefreshedRefFrames = False # Will be set to True later, if applicable

        # For logging processing time
        time_init = time.perf_counter()
        
        if (
            ("update_after_n_triggers" in self.settings) and
            (self.trigger_num >= self.settings["update_after_n_triggers"])
           ):
            # It is time to update the reference period (whilst maintaining phase lock)
            # Set state to "reset" (so we clear things for a new reference period)
            # As part of this reset, trigger_num will be reset
            logger.info(
                "At least {0} triggers have been sent; resetting before switching to adaptive mode.",
                self.settings["update_after_n_triggers"],
            )
            self.state = "reset" 

        pixelArray.metadata["optical_gating_state"] = self.state
        
        if (
            ("save_first_n_frames" in self.settings) and
            (len(self.frames_to_save) < self.settings["save_first_n_frames"])
           ):
            self.frames_to_save.append(pixelArray)
            if len(self.frames_to_save) == self.settings["save_first_n_frames"]:
                ref.save_period(self.frames_to_save, self.settings["reference_sequence_dir"], prefix="VID-")


        if self.state == "sync":
            # Using previously-determined reference period, analyse brightfield frames
            # to determine predicted trigger time for prospective optical gating
            # self.predicted_trigger_time_s.append(
            #     None
            # )  # placeholder - updated inside sync_state

            self.sync_state(pixelArray)

        elif self.state == "reset":
            # Clears reference period and resets frame number
            # Used when determining new period
            self.reset_state()

        elif self.state == "determine":
            # Determine initial reference period and target frame
            self.determine_state(pixelArray)

        elif self.state == "adapt":
            # Determine reference period syncing target frame with original user selection
            self.adapt_state(pixelArray)

        else:
            logger.critical("Unknown state {0}.", self.state)

        # take a note of our processing rate (useful for deciding what camera framerate is viable to use)
        time_fin = time.perf_counter()
        pixelArray.metadata["processing_rate_fps"] = 1 / (
                time_fin - time_init
            )

    def live_phase_interpolation(self): 
        """
        Fluorescence triggers will rarely (if ever) overlap exactly in time with brightfield frames. 
        To achieve an accurate phase estimate at the time a fluorescence image was captured, it is necessary to interpolated between times of known phase. 
        
        This function interpolates phase between the TWO CLOSEST brightfield frames to a given sent trigger time in order to the estimate phase
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
                phaseError = wrappedPhaseAtSentTriggerTime - self.pog_settings["targetSyncPhase"]
            
                # Adjust the phase error to lie in (-pi, pi)
                if phaseError > np.pi:
                    phaseError = phaseError - (2 * np.pi)
                elif phaseError < - np.pi:
                    phaseError = phaseError + (2 * np.pi)
            
            
                self.latestTriggerPredictFrame.metadata["triggerPhaseError"] = phaseError
                self.latestTriggerPredictFrame.metadata["wrappedPhaseAtSentTriggerTime"] = wrappedPhaseAtSentTriggerTime

                logger.info('Live phase interpolation successful! Phase error = {0}', phaseError)
            
                errorThreshold = 0.15
                if abs(phaseError) > errorThreshold:
                    logger.warning('Phase error ({0} radians) has exceeded desired threshold ({1} radians)', phaseError, errorThreshold)
            

    def sync_state(self, pixelArray):
        """ Code to run when in "sync" state
            Synchronising with prospective optical gating for phase-locked triggering.
        """
        logger.debug("Processing frame in prospective optical gating mode.")

        # Gets the phase (in frames) and arrays of SADs between the current frame and the reference sequence
        current_phase, sad, self.pog_settings["drift"] = \
                             pog.identify_phase_with_drift(pixelArray,
                                                           self.pog_settings["ref_frames"],
                                                           self.pog_settings["reference_period"],
                                                           self.pog_settings["drift"])

        # Calculate the unwrapped phase.
        # Normally this is fairly straightforward - calculate (current_phase - previous_phase)
        # and add that to the unwrapped phase of the previous frame.
        # But some care is needed to handle the case where the current phase has just wrapped,
        # or there is a slight backward step from e.g. 0.01 to 2π-0.01.
        if len(self.frame_history) == 0:  # i.e. this is our first frame
            logger.debug("First frame, using current phase as cumulative phase.")
            delta_phase = 0
            phase = current_phase
        else:
            delta_phase = current_phase - self.previous_phase
            # Handle phase wraps in the most sensible way possible
            while delta_phase < -np.pi:
                delta_phase += 2 * np.pi
            phase = self.frame_history[-1].metadata["unwrapped_phase"] + delta_phase

        # Evicts the oldest entry in frame_history if it exceeds the history length that we are meant to be retaining
        # Note: deletion of first list element is potentially a performance issue,
        # although we are hopefully capping the length low enough that it doesn't become a real bottleneck
        if len(self.frame_history) >= self.settings["frame_buffer_length"]:
            del self.frame_history[0]

        # Append PixelArray object to frame_history list with its metadata
        pixelArray.metadata["unwrapped_phase"] = phase
        pixelArray.metadata["sad_min"] = np.argmin(sad)
        self.frame_history.append(pixelArray)

        logger.debug(
            "Current time: {0} s; cumulative phase: {1} (delta:{2:+f}) rad; sad: {3}",
            self.frame_history[-1].metadata["timestamp"],
            self.frame_history[-1].metadata["unwrapped_phase"],
            delta_phase,
            self.frame_history[-1].metadata["sad_min"],
        )

        # === The main purpose of this function: generating synchronization triggers ===
        # If we have at least one period of phase history, have a go at predicting a future trigger time
        # (Note that this prediction can be disabled by enabling "phase_stamp_only" in pog_settings
        this_predicted_trigger_time_s = None
        sendTriggerNow = 0
        if (len(self.frame_history) > self.pog_settings["reference_period"]
            and self.pog_settings["phase_stamp_only"] != True
        ):
            logger.debug("Predicting trigger...")

            # Make a future prediction
            logger.trace("Predicting next trigger.")
            time_to_wait_seconds, heartRateRadsPerSec = pog.predict_trigger_wait(
                pa.get_metadata_from_list(
                    self.frame_history, ["timestamp", "unwrapped_phase", "sad_min"]
                ),
                self.pog_settings,
                fitBackToBarrier=True,
            )
            logger.trace("Time to wait to trigger: {0} s.".format(time_to_wait_seconds))

            this_predicted_trigger_time_s = (
                self.frame_history[-1].metadata["timestamp"] + time_to_wait_seconds
            )

            # If we have a prediction, consider actually sending the trigger
            if time_to_wait_seconds > 0:
                logger.info("Possible trigger after: {0}s", time_to_wait_seconds)
                # Decide whether the current candidate trigger time should actually be used,
                # or whether we should wait for an improved prediction from the next brightfield frame.
                # Note that time_to_wait_seconds might be updated by this call (to change the value to
                # refer to the next heart cycle) if we have already committed to a trigger in the current cycle.
                #
                # JT TODO: I would like to refactor that logic. I think that better belongs
                # in the prediction code itself. More generally, pog_settings should be refactored
                # to just be actual settings, and *state* (e.g. "lastSent") should be encapsulated
                # in a class. That structure would also mesh better with Ross's Kalman code.
                (
                    time_to_wait_seconds,
                    sendTriggerNow
                ) = pog.decide_whether_to_trigger(
                    self.frame_history[-1].metadata["timestamp"],
                    time_to_wait_seconds,
                    self.pog_settings,
                    heartRateRadsPerSec
                )
                if sendTriggerNow != 0:
                    # Actually send the electrical trigger signal
                    logger.success(
                        "Sending trigger (reason: {0}) at time ({1} plus {2}) s",
                        sendTriggerNow,
                        self.frame_history[-1].metadata["timestamp"],
                        time_to_wait_seconds,
                    )
                    # Note that the following call may block on some platforms
                    # (its exact implementation is for the subclass to determine)
                    self.trigger_fluorescence_image_capture(
                        this_predicted_trigger_time_s
                    )

                    self.latestTriggerPredictFrame = self.frame_history[-1]
                    # trigger_sent is a special flag that is *only* present in the dictionary if we did send a trigger
                    self.frame_history[-1].metadata["trigger_sent"] = sendTriggerNow

                    # Update trigger iterator (for adaptive algorithm)
                    self.trigger_num += 1

        # Update PixelArray with predicted trigger time and trigger type
        self.frame_history[-1].metadata[
            "predicted_trigger_time_s"
        ] = this_predicted_trigger_time_s
        self.frame_history[-1].metadata["trigger_type_sent"] = sendTriggerNow
        self.frame_history[-1].metadata["targetSyncPhase"] = self.pog_settings["targetSyncPhase"]
        logger.debug(
            "Current time: {0} s; predicted trigger time: {1} s; trigger type: {2}",
            self.frame_history[-1].metadata["timestamp"],
            self.frame_history[-1].metadata["predicted_trigger_time_s"],
            self.frame_history[-1].metadata["trigger_type_sent"],
        )

        # store this phase now to calculate the delta phase for the next frame
        self.previous_phase = float(current_phase)

        # Computing the phase at the exact time the most recent trigger was acted upon
        self.live_phase_interpolation()

    def reset_state(self):
        """ Code to run when in "reset" state
            Resetting for a new period determination.
            Clears everything required to get a new period.
            Used if the user is not happy with a period choice,
            or before getting a new reference period in the adaptive mode.
        """
        logger.info("Resetting for new period determination.")
        
        self.pog_settings["ref_frames"] = None
        self.ref_buffer = []
        self.period_guesses = []
        
        # lastSent is used as part of the logic in prospective_optical_gating.py
        # TODO: it's not ideal that the reset logic is here but the variable is used in prospective_optical_gating.
        # A future refactor may want to think about tidying that up...
        self.pog_settings["lastSent"] = 0

        # TODO: JT writes: I don't like this logic - I don't feel this is the right place for it.
        # Also, update_after_n_triggers is one reason why we might want to reset the sync,
        # but the user should have the ability to reset the sync through the GUI, or there might
        # be other future reasons we might want to reset the sync (e.g. after each stack).
        # I think this could partly be tidied by making self.state behave more like a proper finite state machine.
        # What I don't like is the fact that the "update_after_n_triggers" logic effectively appears twice.
        # It appears in analyze(), where it may induce a reset, and then it appears again here as a
        # sort of way of figuring out why this reset was initiated in the first place.
        # Not sure yet what the best solution is, but I'm flagging it for a rethink.
        
        if (
            ("update_after_n_triggers" in self.settings) and
            (self.trigger_num >= self.settings["update_after_n_triggers"])
        ):
            # i.e. if adaptive reset trigger_num and get new period
            # automatically phase-locking with the existing period
            logger.info(
                "Switching to adaptive mode.", self.settings["update_after_n_triggers"]
            )
            self.trigger_num = 0
            self.state = "adapt"
        else:
            logger.info("Switching to determine period mode.")
            self.state = "determine"

    def determine_state(self, pixelArray, modeString="determine period"):
        """ Code to run when in "determine" state
            Determine period mode (default behaviour requires user input).
            In this mode we obtain a minimum number of frames, determine a
            period and then return.
            It is assumed that the user (or cli/flask app) then runs the
            user_select_ref_frame function (and updates the state) before running
            analyse again with the new state.
        """
        logger.debug("Processing frame in {0} mode.".format(modeString))

        # Adds new frame to buffer
        self.ref_buffer.append(pixelArray)
        # Impose an upper limit on the buffer length, to protect against performance degradation
        # in cases where we are not succeeding in identifying a period
        # Note: deletion of first list element is potentially a performance issue,
        # although we are hopefully capping the length low enough that it doesn't become a real bottleneck
        ref_buffer_duration = (self.ref_buffer[-1].metadata["timestamp"]
                               - self.ref_buffer[0].metadata["timestamp"])
        if (
            ("min_heart_rate_hz" in self.settings) and
            (ref_buffer_duration > 1.0/self.settings["min_heart_rate_hz"])
           ):
            logger.debug("Trimming buffer to duration {0}".format(1.0/self.settings["min_heart_rate_hz"]))
            del self.ref_buffer[0]

        # Calculate period from determine_reference_period.py
        logger.info("Attempting to determine new reference period.")
        ref_frames, period_to_use = ref.establish(
            self.ref_buffer, self.period_guesses, self.pog_settings
        )

        if ref_frames is not None:
            self.pog_settings.set_reference_frames(ref_frames, period_to_use)

            if "reference_sequence_dir" in self.settings:
                # Save the reference sequence to disk, for debug purposes
                ref.save_period(self.pog_settings["ref_frames"], self.settings["reference_sequence_dir"])
            logger.success("Period determined.")
            self.justRefreshedRefFrames = True   # Flag that a slow action took place

            if self.automatic_target_frame_selection:
                logger.info(
                    "Period determined and target frame automatically selected; switching to prospective optical gating mode."
                )
                # Automatically switch to the "sync" state, using the default reference frame.
                # The user might choose to change the reference frame later, via a GUI.
                self.start_sync_with_ref_frame(None)
                self.state = "sync"
            else:
                # If we aren't using the automatically determined period
                # We raise the stop flag, which returns the current state to the user/app
                # The user/app can then select a target frame
                # The user/app will also need to call the adaptive system
                # see user_select_ref_frame()
                # **** JT update this comment?
                # This process is a bit weird - perhaps it's time to think about what a proper state machine would look like?
                self.stop = 'select'


    def adapt_state(self, pixelArray):
        """ Code to run when in "adapt" state.
            Adaptive prospective optical gating mode
            i.e. update reference sequence, while maintaining the same phase-lock.
            In this mode we determine a new period and then align with
            previous periods using an adaptive algorithm.
        """

        # Start by calling through to determine_state() to establish a new reference sequence
        # JT: I think this is backwards. I think it makes more sense for determine_state to know if we should be being adaptive,
        #     and doing the adapt logic if required. Otherwise we set a target frame and then immediately change it to something else
        self.determine_state(pixelArray, modeString="adaptive optical gating")

        if self.pog_settings["ref_frames"] is not None:
            # First, spot if we are setting up the OGA process for the first time.
            # If so, we need to set oga_reference_value
            if self.sequence_history is None:
                relTargetPos = self.pog_settings["referenceFrame"] / self.pog_settings["reference_period"]
                self.pog_settings["oga_reference_value"] = self.pog_settings["oga_resampled_period"] * relTargetPos
            
            # Align the current reference sequence relative to previous ones (adaptive update)
            # Note that the ref_seq_phase parameter for process_sequence is in units
            (
                self.sequence_history,
                self.period_history,
                self.drift_history,
                self.shift_history,
                self.global_solution,
                self.pog_settings["referenceFrame"],
            ) = oga.process_sequence(
                self.pog_settings["ref_frames"],
                self.pog_settings["reference_period"],
                self.pog_settings["drift"],
                sequence_history=self.sequence_history,
                period_history=self.period_history,
                drift_history=self.drift_history,
                shift_history=self.shift_history,
                global_solution=self.global_solution,
                max_offset=3,
                ref_seq_id=0,
                ref_seq_phase=self.pog_settings["oga_reference_value"],
                resampled_period=self.pog_settings["oga_resampled_period"]
            )
            self.justRefreshedRefFrames = True   # Flag that a slow action took place
            logger.success(
                "Reference period updated. New period of length {0} with reference frame at {1}",
                self.pog_settings["reference_period"],
                self.pog_settings["referenceFrame"],
            )

            # Switch back to the sync state
            logger.info(
                "Period updated and adaptive phase-lock successful; switching back to prospective optical gating mode."
            )
            self.state = "sync"

    def start_sync_with_ref_frame(self, ref_frame_number, barrier=None):
        defaultRef, defaultBarrier = pog.pick_target_and_barrier_frames(self.pog_settings["ref_frames"],
                                                                        self.pog_settings["reference_period"])
        if ref_frame_number is None:
            ref_frame_number = defaultRef
        if barrier is None:
            barrier = defaultBarrier
        self.pog_settings.set_reference_and_barrier_frame(ref_frame_number, barrier)
        
        # Turn recording back on for rest of run
        self.stop = False
        # Turn automatic target frames on for future adaptive updates
        self.automatic_target_frame_selection = True
        # Switch to "sync" state, in which we send camera triggers
        logger.info(
                    "Starting sync. Period determined and target frame has been selected by the user/app; switching to prospective optical gating mode."
                    )
        self.state = "sync"

    def user_select_ref_frame(self, ref_frame_number=None):
        """Prompts the user to select the target frame from a one-period set of reference frames"""
        if ref_frame_number is None:
            # For now it is a simple command line interface (which is not helpful at all)
            ref_frame_number = int(
                input(
                    "Please select a frame between 0 and "
                    + str(len(self.pog_settings["ref_frames"]) - 1)
                    + "\nOr enter -1 to select a new period.\n"
                )
            )

        if ref_frame_number < 0:
            # User wants to select a new period. Users can use their creative side by selecting any negative number.
            logger.success(
                           "User has asked for a new period to be determined, resetting before switching to period determination mode."
                           )
            self.state = "reset"
        else:
            # Commit to using this reference frame
            self.start_sync_with_ref_frame(ref_frame_number)

    def trigger_fluorescence_image_capture(self, trigger_time_s):
        """
        As this is the base server, this function just outputs a log that a trigger would have been sent.
        """
        logger.success("A fluorescence image would be triggered now.")

    def plot_triggers(self, outfile="triggers.png"):
        """
        Plot the phase vs. time sawtooth line with trigger events.
        """

        # get trigger times from predicted triggers time and trigger types sent (e.g. not 0)
        sent_trigger_times = pa.get_metadata_from_list(
            self.frame_history, "predicted_trigger_time_s"
        )[pa.get_metadata_from_list(self.frame_history, "trigger_type_sent") > 0]

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
            np.full(
                max(len(sent_trigger_times), 0), self.pog_settings["targetSyncPhase"],
            ),
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
            pa.get_metadata_from_list(self.frame_history, "wrappedPhaseAtSentTriggerTime", onlyIfKeyPresent="trigger_sent"), 
            bins = np.arange(0, 2 * np.pi, 0.01), 
            color = "green", 
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
            onlyIfKeyPresent="trigger_sent"
        )
        plt.hist(
            phaseErrorList,
            bins = np.arange(np.min(phaseErrorList), np.max(phaseErrorList) + 0.1,  0.03), 
            color = "green", 
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
            color = 'red'
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
