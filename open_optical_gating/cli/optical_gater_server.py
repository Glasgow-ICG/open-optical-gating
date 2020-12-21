"""Parent Open Optical Gating Class"""

# Python imports
import sys
import json
import time

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
from . import parameters as parameters

logger.remove()
logger.add(sys.stderr, level="WARNING")
# logger.add("oog_{time}.log", level="DEBUG")
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
            settings - a dictionary of settings (see default_settings.json)
        """

        # store the whole settings dict
        # we occasionally store some of this information elsewhere too
        # that's not ideal but works for now
        self.settings = settings
        # NOTE: there is also self.pog_settings, be careful of this

        if ref_frames is not None:
            logger.success("Using existing reference frames...")
        self.ref_frames = ref_frames
        self.ref_frame_period = ref_frame_period
        logger.success("Initialising internal parameters...")
        self.initialise_internal_parameters()
        self.automatic_target_frame = True
        self.justRefreshedRefFrames = False

    def initialise_internal_parameters(self):
        """Defines all internal parameters not already initialised"""
        # Defines an empty list to store past frames with timestamp, phase and argmin(sad) metadata
        self.frame_history = []
        self.pixel_dtype = "uint8"

        # Variables for adaptive algorithm
        self.trigger_num = 0
        self.sequence_history = None
        self.period_history = None
        self.shift_history = None
        self.drift_history = None

        # TODO: JT writes: this seems as good a place as any to flag the fact that I don't think barrier frames are being implemented properly.
        # There is a call to determine_barrier_frames, but I don't think the *value* for the barrier frame parameter is ever computed, is it?
        # It certainly isn't when using existing reference frames. This seems like an important missing bit of code.
        # I think it just defaults to 0 when the settings are initialised, and stays that way.

        # Start by acquiring a sequence of reference frames, unless we have been provided with them
        if self.ref_frames is None:
            logger.info(
                "No reference frames found, resetting before switching to determine period mode."
            )
            self.state = "reset"
            self.pog_settings = parameters.initialise(
                framerate=self.settings["brightfield_framerate"]
            )
        else:
            logger.info(
                "Using existing reference frames with integer period, setting to prospective optical gating mode."
            )
            self.state = "sync"
            if self.ref_frame_period is None:
                # Deduce an integer reference period from the reference frames we were provided with.
                # This is just a legacy mode - caller who constructed this object should really have provided a reference period
                rp = self.ref_frames.shape[0] - 2 * self.settings["NumExtraRefFrames"]
            else:
                # Use the reference period provided when this object was constructed.
                rp = self.ref_frame_period

            self.pog_settings = parameters.initialise(
                framerate=self.settings["brightfield_framerate"], reference_period=rp,
            )
            self.pog_settings = pog.determine_barrier_frames(self.pog_settings)

        # Start experiment timer
        self.initial_process_time_s = time.time()

        # Flag for interrupting the program at key points
        # E.g. when user-input is needed
        # It is assumed that the user/app controls what this interaction is
        self.stop = False

    def analyze_pixelarray(self, pixelArray):
        """ Method to analyse each frame as they are captured by the camera.
            The documentation explains that this must be fast, since it is running within the encoder's callback,
            and so must return before the next frame is produced.
            Essentially this method just calls through to another appropriate method, based on the current value of the state attribute."""
        logger.debug(
            "Analysing frame with timestamp: {0}s", pixelArray.metadata["timestamp"],
        )
        self.justRefreshedRefFrames = False # Will be set to True later, if applicable

        # For logging processing time
        time_init = time.perf_counter()

        # TODO: These lines need to be moved into the eventual
        # pi_optical_gater analyze (inherited from picamera) method
        # If we're passed a colour image, take the first channel (Y; luma)
        # if isinstance(frame, pa.PixelArray):
        #     logger.info('PixelArray object passed to analyze.')
        #     pixelArray = frame
        # elif isinstance(frame, np.ndarray) and len(pixelArray.shape) == 3:
        #     logger.info('Colour frame (LUV) passed to analyze, only using luma (Y) channel in PixelArray object.')
        #     pixelArray = pa.PixelArray(frame[:, :, 0], metadata={'timestamp':time_init})
        # elif isinstance(frame, np.ndarray) and len(pixelArray.shape) == 2:
        #     logger.info('Greyscale frame passed to analyze, converting to PixelArray object.')
        #     pixelArray = pa.PixelArray(frame, metadata={'timestamp':time_init})
        # else:
        #     logger.critical('Frame of unknown type passed to analyze.')

        if self.trigger_num >= self.settings["update_after_n_triggers"]:
            # It is time to update the reference period (whilst maintaining phase lock)
            # Set state to "reset" (so we clear things for a new reference period)
            # As part of this reset, trigger_num will be reset
            logger.info(
                "At least {0} triggers have been sent; resetting before switching to adaptive mode.",
                self.settings["update_after_n_triggers"],
            )
            self.state = "reset"

        pixelArray.metadata["optical_gating_state"] = self.state

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

        # take a note of our processing rate (useful for deciding what framerate to set)
        time_fin = time.perf_counter()
        pixelArray.metadata["processing_rate_fps"] = 1 / (
                time_fin - time_init
            )

    def sync_state(self, pixelArray):
        """ Code to run when in "sync" state
            Synchronising with prospective optical gating for phase-locked triggering.
        """
        logger.debug("Processing frame in prospective optical gating mode.")

        # Gets the phase (in frames) and arrays of SADs between the current frame and the referencesequence
        currentPhaseInFrames, sad, self.pog_settings = pog.phase_matching(
            pixelArray, self.ref_frames, settings=self.pog_settings
        )
        logger.trace(sad)

        # Convert phase to 2pi base
        current_phase = (
            2
            * np.pi
            * (currentPhaseInFrames - self.pog_settings["numExtraRefFrames"])
            / self.pog_settings["reference_period"]
        )  # rad

        # Calculate cumulative phase (phase) from delta phase (current_phase - last_phase)
        if len(self.frame_history) == 0:  # i.e. first frame
            logger.debug("First frame, using current phase as cumulative phase.")
            delta_phase = 0
            phase = current_phase
            self.last_phase = current_phase
        else:
            delta_phase = current_phase - self.last_phase
            while delta_phase < -np.pi:
                delta_phase += 2 * np.pi
            phase = self.frame_history[-1].metadata["unwrapped_phase"] + delta_phase
            self.last_phase = current_phase

        # Evicts the oldest entry in frame_history if it exceeds the history length that we are meant to be retaining
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

        # If we have at least one period of phase history, have a go at predicting a future trigger time
        # (Note that this prediction can be disabled by enabling "phase_stamp_only" in pog_settings
        this_predicted_trigger_time_s = None
        sendTriggerNow = 0
        if (len(self.frame_history) > self.pog_settings["reference_period"]
            and self.pog_settings["phase_stamp_only"] != True
        ):
            logger.debug("Predicting trigger...")

            # TODO: JT writes: this seems as good a place as any to highlight the general issue that the code is not doing a great job of precise timing.
            # It determines a delay time before sending the trigger, but then executes a bunch more code.
            # Oh and, more importantly, that delay time is then treated relative to “current_time_s”, which is set *after* doing the phase-matching.
            # That is going to reduce accuracy and precision, and also makes me even more uncomfortable in terms of future-proofing.
            # I think it would be much better to pass around absolute times, not deltas.

            # Gets the trigger response
            logger.trace("Predicting next trigger.")
            time_to_wait_seconds = pog.predict_trigger_wait(
                pa.get_metadata_from_list(
                    self.frame_history, ["timestamp", "unwrapped_phase", "sad_min"]
                ),
                self.pog_settings,
                fitBackToBarrier=True,
            )
            logger.trace("Time to wait: {0} s.".format(time_to_wait_seconds))
            # frame_history is an nx3 array of [timestamp, phase, argmin(SAD)]
            # phase (i.e. frame_history[:,1]) should be cumulative 2Pi phase
            # targetSyncPhase should be in [0,2pi]

            this_predicted_trigger_time_s = (
                self.frame_history[-1].metadata["timestamp"] + time_to_wait_seconds
            )

            # Captures the image
            if time_to_wait_seconds > 0:
                logger.info("Possible trigger after: {0}s", time_to_wait_seconds)

                (
                    time_to_wait_seconds,
                    sendTriggerNow,
                    self.pog_settings,
                ) = pog.decide_trigger(
                    self.frame_history[-1].metadata["timestamp"],
                    time_to_wait_seconds,
                    self.pog_settings,
                )
                if sendTriggerNow != 0:
                    logger.success(
                        "Sending trigger (reason: {0}) at time ({1} plus {2}) s",
                        sendTriggerNow,
                        self.frame_history[-1].metadata["timestamp"],
                        time_to_wait_seconds,
                    )
                    # Trigger only
                    self.trigger_fluorescence_image_capture(
                        this_predicted_trigger_time_s
                    )

                    # Update trigger iterator (for adaptive algorithm)
                    self.trigger_num += 1

        # Update PixelArray with predicted trigger time and trigger type
        self.frame_history[-1].metadata[
            "predicted_trigger_time_s"
        ] = this_predicted_trigger_time_s
        self.frame_history[-1].metadata["trigger_type_sent"] = sendTriggerNow
        logger.debug(
            "Current time: {0} s; predicted trigger time: {1} s; trigger type: {2}",
            self.frame_history[-1].metadata["timestamp"],
            self.frame_history[-1].metadata["predicted_trigger_time_s"],
            self.frame_history[-1].metadata["trigger_type_sent"],
        )

        # store this phase now to calculate the delta phase for the next frame
        self.last_phase = float(current_phase)

    def reset_state(self):
        """ Code to run when in "reset" state
            Resetting for a new period determination.
            Clears everything required to get a new period.
            Used if the user is not happy with a period choice,
            or before getting a new reference period in the adaptive mode.
        """
        logger.info("Resetting for new period determination.")
        self.ref_frames = None
        self.ref_buffer = []
        self.period_guesses = []

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
            self.settings["update_after_n_triggers"] > 0
            and self.trigger_num >= self.settings["update_after_n_triggers"]
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

        # Calculate period from determine_reference_period.py
        logger.info("Attempting to determine new reference period.")
        self.ref_frames, self.pog_settings = ref.establish(
            self.ref_buffer, self.period_guesses, self.pog_settings
        )

        if self.ref_frames is not None:
            # We were provided with ref_frames as a list, and this is helpful because it means we could still access
            # the PixelArray metadata at this point if we wished.

            # However, long-term we want to store a 3D array because that is what oga expects to work with.
            # We therefore make that conversion here
            self.ref_frames = np.array(self.ref_frames)

            # Automatically select a target frame and barrier
            # This can be overriden by the user/controller later
            self.pog_settings = pog.pick_target_and_barrier_frames(
                self.ref_frames, self.pog_settings
            )

            # Determine barrier frames
            self.pog_settings = pog.determine_barrier_frames(self.pog_settings)

            # Save the period
            ref.save_period(self.ref_frames, self.settings["period_dir"])
            logger.success("Period determined.")
            self.justRefreshedRefFrames = True   # Flag that a slow action took place

            if self.automatic_target_frame:
                logger.info(
                    "Period determined and target frame automatically selected; switching to prospective optical gating mode."
                )
                # Automatically switch to the "sync" state, using the default reference frame.
                # The user is expected to change the reference frame later, via a GUI, if they wish to
                self.start_sync_with_ref_frame(self.pog_settings["referenceFrame"])
                self.state = "sync"
            else:
                # If we aren't using the automatically determined period
                # We raise the stop flag, which returns the current state to the user/app
                # The user/app can then select a target frame
                # The user/app will also need to call the adaptive system
                # see user_select_ref_frame()
                self.stop = True


    def adapt_state(self, pixelArray):
        """ Code to run when in "adapt" state.
            Adaptive prospective optical gating mode
            i.e. update reference sequence, while maintaining the same phase-lock.
            In this mode we determine a new period and then align with
            previous periods using an adaptive algorithm.
        """

        # Start by calling through to determine_state() to establish a new reference sequence
        self.determine_state(pixelArray, modeString="adaptive optical gating")

        if self.ref_frames is not None:
            # Align the current reference sequence relative to previous ones (adaptive update)
            (
                self.sequence_history,
                self.period_history,
                self.drift_history,
                self.shift_history,
                self.global_solution,
                self.target,
            ) = oga.process_sequence(
                self.ref_frames,
                self.pog_settings["reference_period"],
                self.pog_settings["drift"],
                sequence_history=self.sequence_history,
                period_history=self.period_history,
                drift_history=self.drift_history,
                shift_history=self.shift_history,
                global_solution=self.global_solution,
                max_offset=3,
                ref_seq_id=0,
                ref_seq_phase=self.pog_settings["referenceFrame"],
            )
            self.justRefreshedRefFrames = True   # Flag that a slow action took place
            self.pog_settings = parameters.update(
                self.pog_settings,
                referenceFrame=(
                    self.pog_settings["reference_period"] * self.target / 80
                )  # TODO 80 here should be a user-defined variable; we tend not to change it but let's give them the option
                % self.pog_settings["reference_period"],
            )
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

    def start_sync_with_ref_frame(self, ref_frame_number):
        self.pog_settings = parameters.update(self.pog_settings, referenceFrame=ref_frame_number)
        # add to periods history for adaptive updates
        (
         self.sequence_history,
         self.period_history,
         self.drift_history,
         self.shift_history,
         self.global_solution,
         self.target,
         ) = oga.process_sequence(
                                  self.ref_frames,
                                  self.pog_settings["reference_period"],
                                  self.pog_settings["drift"],
                                  max_offset=3,
                                  ref_seq_id=0,
                                  ref_seq_phase=ref_frame_number,
                                  )

        # Turn recording back on for rest of run
        self.stop = False
        # Turn automatic target frames on for future adaptive updates
        self.automatic_target_frame = True
        # Switch to "sync" state, in which we send camera triggers
        logger.info(
                    "Period determined and target frame has been selected by the user/app; switching to prospective optical gating mode."
                    )
        self.state = "sync"

    def user_select_ref_frame(self, ref_frame_number=None):
        """Prompts the user to select the target frame from a one-period set of reference frames"""
        if ref_frame_number is None:
            # For now it is a simple command line interface (which is not helpful at all)
            ref_frame_number = int(
                input(
                    "Please select a frame between 0 and "
                    + str(len(self.ref_frames) - 1)
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

    def trigger_fluorescence_image_capture(self, delay):
        """As this is the base server, this function just outputs a log that a trigger would have been sent."""
        logger.success("A fluorescence image would be triggered now.")

    def plot_triggers(self, outfile="triggers.png"):
        """Plot the phase vs. time sawtooth line with trigger events."""

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
        # Add labels etc
        # x_1, x_2, _, y_2 = plt.axis()
        # plt.axis((x_1, x_2, 0, y_2 * 1.1))
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (rad)")

        # Saves the figure
        plt.savefig(outfile)
        plt.show()

    def plot_accuracy(self, outfile="accuracy.png"):
        """Plot the target phase and adjusted real phase of trigger events."""
        wrapped_phase = pa.get_metadata_from_list(
            self.frame_history, "unwrapped_phase"
        ) % (2 * np.pi)

        # get trigger times from predicted triggers time and trigger types sent (e.g. not 0)
        sent_trigger_times = pa.get_metadata_from_list(
            self.frame_history, "predicted_trigger_time_s"
        )[pa.get_metadata_from_list(self.frame_history, "trigger_type_sent") > 0]

        triggeredPhase = []
        for i in range(len(sent_trigger_times)):

            triggeredPhase.append(
                wrapped_phase[
                    (
                        np.abs(
                            pa.get_metadata_from_list(self.frame_history, "timestamp")
                            - sent_trigger_times[i]
                        )
                    ).argmin()
                ]
            )

        plt.figure()
        plt.title("Frequency density of triggered phase")
        bins = np.arange(0, 2 * np.pi, 0.1)
        plt.hist(triggeredPhase, bins=bins, color="g", label="Triggered phase")
        x_1, x_2, y_1, y_2 = plt.axis()
        plt.plot(
            np.full(2, self.pog_settings["targetSyncPhase"]),
            (y_1, y_2),
            "r-",
            label="Target phase",
        )
        plt.xlabel("Triggered phase (rad)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.axis((x_1, x_2, y_1, y_2))

        plt.tight_layout()
        plt.savefig(outfile)
        plt.show()

    def plot_prediction(self, outfile="prediction.png"):
        plt.figure()
        plt.title("Predicted Trigger Times")
        plt.plot(
            pa.get_metadata_from_list(self.frame_history, "timestamp"),
            pa.get_metadata_from_list(self.frame_history, "predicted_trigger_time_s"),
        )
        # Add labels etc
        plt.xlabel("Time (s)")
        plt.ylabel("Prediction (s)")

        # Saves the figure
        plt.savefig(outfile)
        plt.show()

    def plot_running(self, outfile="running.png"):
        plt.figure()
        plt.title("Frame processing rate")
        plt.plot(
            pa.get_metadata_from_list(self.frame_history, "timestamp"),
            pa.get_metadata_from_list(self.frame_history, "processing_rate_fps"),
        )
        # Add labels etc
        plt.xlabel("Time (s)")
        plt.ylabel("Processing rate (fps)")

        # Saves the figure
        plt.savefig(outfile)
        plt.show()
