"""Main CLI Open Optical Gating System"""

# Python imports
import sys
import json
import time

# Module imports
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

try:
    import picamera
    from picamera.array import PiYUVAnalysis
except:
    # JT: Temporary hack for now, creating a dummy PiYUVAnalysis class to enable the code below to compile in non-Pi environments
    class PiYUVAnalysis:
        def __init__(self, camera):
            pass


# Fastpins module
import fastpins as fp

# Optical Gating Alignment module
import optical_gating_alignment.optical_gating_alignment as oga

# Local imports
import open_optical_gating.cli.determine_reference_period as ref
import open_optical_gating.cli.prospective_optical_gating as pog
import open_optical_gating.cli.parameters as parameters
import open_optical_gating.cli.stage_control_functions as scf

logger.remove()
logger.add(sys.stderr, level="SUCCESS")
logger.enable("open_optical_gating")

# TODO there are one or two places where settings should be updated, e.g. user-bound limits
# TODO create a time-stamped copy of the settings file after this
# TODO create a time-stamped log somewhere


class OpticalGater(PiYUVAnalysis):
    """Custom class to convert and analyse Y (luma) channel of each YUV frame.
    Extends the picamera.array.PiYUVAnalysis class, which has a stub method called analyze that is overidden here.

    # TODO: JT writes: I think these should be short strings instead of numbers, to be suitably descriptive of what the mode is, rather than just magic numbers:
    self.state has several modes:
       0 - run prospective gating mode (phase locked triggering)
       1 - re-initialise (clears for mode 2)
       2 - get period mode (requires user input)
       3 - adaptive mode (update period but maintain phase lock)

    """

    def __init__(
        self, camera=None, settings=None, ref_frames=None, ref_frame_period=None
    ):
        """Function inputs:
            camera - the raspberry picam PiCamera object
            settings - a dictionary of settings (see default_settings.json)
        """

        # store the whole settings dict
        # we occasionally store some of this information elsewhere too
        # that's not ideal but works for now
        self.settings = settings
        # NOTE: there is also self.pog_settings, be careful of this
        # TODO: JT writes: UGH! Who has responsibility for the settings object? What is the distinction between settings and pog_settings?
        #                  Surely settings["frame_buffer_length"] should be in pog_settings?
        #                  Maybe hold off doing anything about this until after the refactors I believe are needed - but we should make sure this is tidied and clarified eventually

        logger.success("Setting camera settings...")
        if camera is not None and not isinstance(camera, str):
            super(OpticalGater, self).__init__(camera)
            self.setup_camera(camera)
        elif isinstance(camera, str):
            self.load_data(camera)
        else:
            logger.critical("No camera or data found.")
        logger.success("Initialising triggering hardware...")
        self.init_hardware()
        if ref_frames is not None:
            logger.success("Using existing reference frames...")
        self.ref_frames = ref_frames
        self.ref_frame_period = ref_frame_period
        logger.success("Initialising internal parameters...")
        self.initialise_internal_parameters()

    def setup_camera(self, camera):
        """Initialise and apply camera-related settings."""
        self.frame_num = self.settings["frame_num"]
        self.width, self.height = camera.resolution
        self.framerate = camera.framerate
        self.camera = camera

    def load_data(self, filename):
        """Place holder function for loading data in emulator.
        In this instance this function is just a catch if the user passes a
        string rather than a picamera instance when initialising the
        OpticalGater object.
        In the emulate module, a different definition of OpticalGater is
        created, inheriting from this definition, and that definition uses
        this function to read a whole file in to memory.
        """
        logger.critical("No camera found in live mode ({0}).", filename)

    def initialise_internal_parameters(self):
        """Defines all internal parameters not already initialised"""
        # Defines the arrays for sad and frame_history (which contains period, timestamp and argmin(sad))
        self.frame_history = np.zeros((self.settings["frame_buffer_length"], 3))
        self.dtype = "uint8"

        # Variables for adaptive algorithm
        # Should be compatible with standard adaptive algorithm
        #  (Nat Comms paper) and dynamic algorithm (unpublished, yet)
        # if update_after_n_triggers is 0, turns adaptive mode off   # TODO: JT writes: what does this comment have to do with the surrounding code? Does it belong somewhere else, perhaps?
        self.trigger_num = 0
        self.sequence_history = []
        self.period_history = []
        self.shift_history = []
        self.drift_history = []

        # Sets ouput mode for normal trigger or special Glasgow mode
        if self.settings["trigger_mode"] == "5V_BNC_Only":
            self.moveStagesAfterTrigger = False
        else:
            self.moveStagesAfterTrigger = True

        # TODO: JT writes: this seems as good a place as any to flag the fact that I don't think barrier frames are being implemented properly.
        # There is a call to determine_barrier_frames, but I don't think the *value* for the barrier frame parameter is ever computed, is it?
        # It certainly isn't when using existing reference frames. This seems like an important missing bit of code.
        # I think it just defaults to 0 when the settings are initialised, and stays that way.

        # Start by acquiring a sequence of reference frames, unless we have been provided with them
        if self.ref_frames is None:
            logger.info("No reference frames found, switching to 'get period' mode.")
            self.state = 1
            self.pog_settings = parameters.initialise(framerate=self.framerate)
        else:
            logger.info("Using existing reference frames with integer period.")
            self.state = 0
            if self.ref_frame_period is None:
                # Deduce an integer reference period from the reference frames we were provided with.
                # This is just a legacy mode - caller who constructed this object should really have provided a reference period
                rp = self.ref_frames.shape[
                    0
                ]  # TODO: JT writes: what about padding!? I think this does not take proper account of numExtraRefFrames, does it?
            else:
                # Use the reference period provided when this object was constructed.
                rp = self.ref_frame_period

            self.pog_settings = parameters.initialise(
                framerate=self.framerate, referencePeriod=rp
            )
            self.pog_settings = pog.determine_barrier_frames(self.pog_settings)

        # Start experiment timer
        self.initial_process_time = time.time()

        # Defines variables and objects used for plotting
        self.timestamp = []
        self.phase = []
        self.process_time = []
        self.trigger_times = []

        # Flag for interrupting the program at key points
        # E.g. when user-input is needed
        # It is assumed that the user/app controls what this interaction is
        self.stop = False

    def init_hardware(self):
        """Function that initialises various hardware I/O interfaces (pins for triggering laser and fluorescence camera, along with the USB for controlling the Newport stages)
        Function inputs:
            laser_trigger_pin = the GPIO pin number connected to fire the laser
            fluorescence_camera_pins = an array of 3 pins used to interact with the fluoresence camera
                                            (trigger, SYNC-A, SYNC-B)
            self.settings["usb_stages"] = a list containing the information used to set up the usb for controlling the Newport stages
                                    (USB address (str),timeout (flt), baud rate (int), byte size (int), parity (char), stop bits (int), xonxoff (bool))
                                    TODO: JT writes: no, this seems to be a dictionary, not a list...
        When this function return, self.usb_serial will be one of:
            0 - if no failure and no usb stage
            1 - if fastpins fails
            2 - if laser pin fails
            3 - if camera pins fail
            4 - if usb stages fail
            serial object - if no failure and usb stages desired
        """
        # TODO update this docstring
        self.usb_serial = (
            None  # default - no stages but carry on as if everything else worked
        )

        # Initialises fastpins module
        try:
            logger.debug("Initialising fastpins...")
            fp.init()
        except Exception as inst:
            logger.critical("Error setting up fastpins module. {0}", inst)

        # Sets up laser trigger pin
        laser_trigger_pin = self.settings["laser_trigger_pin"]
        if laser_trigger_pin is not None:
            try:
                logger.debug("Initialising BNC trigger pin... (Laser/Trigger Box)")
                # PUD resistor needs to be specified but will be ignored in setup
                fp.setpin(laser_trigger_pin, 1, 0)
            except Exception as inst:
                logger.critical("Error setting up laser pin. {0}", inst)

        # Sets up fluorescence camera pins
        # TODO: JT writes: why not make these a [sub-]dictionary, so we don't have to keep track of what [0], [1], and [2] represent?
        fluorescence_camera_pins = self.settings[
            "fluorescence_camera_pins"
        ]  # Trigger, SYNC-A, SYNC-B
        if fluorescence_camera_pins is not None:
            try:
                logger.debug("Initialising fluorescence camera pins...")
                fp.setpin(fluorescence_camera_pins[0], 1, 0)  # Trigger
                fp.setpin(fluorescence_camera_pins[1], 0, 0)  # SYNC-A
                fp.setpin(fluorescence_camera_pins[2], 0, 0)  # SYNC-B
                self.duration = 1e3  # TODO relate to settings
                self.edge_trigger = True  #  TODO relate to settings     # TODO: JT writes: flagging this again as definitely belonging in settings!
            except Exception as inst:
                logger.critical("Error setting up fluorescence camera pins. {0}", inst)

        # Sets up USB for Newport stages (Glasgow)
        # self.settings["usb_stages"] will be either None (no stages) or a dict of:
        # "name", "timeout", "baud_rate", "data_bits", "parity", "x_on_off"
        # "current_position", "encoding", "increment", "negative_limit", "plane_address", "positive_limit", "terminators"
        if self.settings["usb_stages"] is not None:
            # init_stage function needs to a tuple of
            # (address, timeout, baud_rate, data_bits, parity, x_on_off, True)
            # TODO: ABD to work out what True is for!
            # TODO: JT writes: this is just wrong. Check init_stage parameters, this code is passing x_on_off as the stopbits parameter!
            #                  Why not just pass the settings dictionary, and have init_stage parse it? That will avoid bugs like this.
            try:
                logger.debug("Initialising USB stages...")
                self.usb_serial = scf.init_stage(
                    (
                        self.settings["usb_stages"]["name"],
                        self.settings["usb_stages"]["timeout"],
                        self.settings["usb_stages"]["baud_rate"],
                        self.settings["usb_stages"]["data_bits"],
                        self.settings["usb_stages"]["parity"],
                        self.settings["usb_stages"]["x_on_off"],
                        True,
                    )
                )
            except Exception as inst:
                logger.critical("Error setting up stages. {0}", inst)
        else:
            logger.debug("No USB stage information provided.")

        # Check that usb_serial was successfully set up
        if self.usb_serial is not None:
            logger.success("Serial stages found; getting z-stage stack bounds.")
            # Defines variables for USB serial stage commands
            self.settings["usb_stages"]["terminator"] = chr(
                self.settings["usb_stages"]["terminators"][0]
            ) + chr(self.settings["usb_stages"]["terminators"][1])

            # Sets up stage to receive future input
            # TODO integrate into web app
            (
                self.negative_limit,
                self.positive_limit,
                self.current_position,
            ) = scf.set_user_stage_limits(
                self.usb_serial,
                self.settings["usb_stages"]["plane_address"],
                self.settings["usb_stages"]["encoding"],
                self.settings["usb_stages"]["terminator"],
            )

    def analyze(self, pixelArray):
        """ Method to analyse each frame as they are captured by the camera.
            The documentation explains that this must be fast, since it is running within the encoder's callback,
            and so must return before the next frame is produced.
            Essentially this method just calls through to another appropriate method, based on the current value of the state attribute."""
        logger.debug("Analysing frame.")

        # For logging processing time
        time_init = time.time()

        # If we're passed a colour image, take the first channel (Y; luma)
        if len(pixelArray.shape) == 3:
            pixelArray = pixelArray[:, :, 0]

        if self.trigger_num >= self.settings["update_after_n_triggers"]:
            # It is time to update the reference period (whilst maintaining phase lock)
            # Set state to 1 (so we clear things for a new reference period)
            # As part of this reset, trigger_num will be reset
            self.state = 1

        # Ensures stage is always within user defined limits (only needed for RPi-controlled stages)
        stages_safe = (self.usb_serial is None) or (
            self.current_position <= self.positive_limit
            and self.current_position >= self.negative_limit
        )
        # TODO: JT writes: this test "if stages_safe" is pretty counter-intuitive at this point.
        # Needs a comment to explain why the stages have anything at all to do with this!
        # That said, after my proposed refactor this should not need to appear here,
        # so let's just leave this comment in place as a reminder, for now.
        # Related: pog_state() updates the stages, which would also be tackled in my refactor.
        logger.debug("Stages safe? {0}", stages_safe)
        if stages_safe:
            if self.state == 0:
                # Using previously-determined reference peiod, analyse brightfield frames
                # to determine predicted trigger time for prospective optical gating
                (trigger_response, current_phase, current_time) = self.pog_state(
                    pixelArray
                )

                # Logs results and processing time
                time_fin = time.time()
                self.timestamp.append(current_time)
                self.phase.append(current_phase)
                self.process_time.append(time_fin - time_init)

                return trigger_response, current_phase, current_time

            elif self.state == 1:
                # Clears reference period and resets frame number
                # Used when determining new period
                self.clear_state()

            elif self.state == 2:
                # Determine initial reference period and target frame
                self.get_period_state(pixelArray)

            elif self.state == 3:
                # Determine reference period syncing target frame with original user selection
                self.update_reference_sequence(pixelArray)

            else:
                logger.critical("Unknown state {0}.", self.state)

        # Default return - all None
        return (None, None, None)

    def pog_state(self, frame):
        """State 0 - run prospective optical gating mode (phase locked triggering)."""
        logger.debug("Processing frame in prospective optical gating mode.")

        # Gets the phase (in frames) and arrays of SADs between the current frame and the referencesequence
        currentPhaseInFrames, sad, self.pog_settings = pog.phase_matching(
            frame, self.ref_frames, settings=self.pog_settings
        )
        logger.trace(sad)

        # Convert phase to 2pi base
        current_phase = (
            2
            * np.pi
            * (currentPhaseInFrames - self.pog_settings["numExtraRefFrames"])
            / self.pog_settings["referencePeriod"]
        )  # rad

        # Gets the current timestamp in milliseconds
        current_time = (
            time.time() - self.initial_process_time
        ) * 1000  # Converts time into milliseconds

        # Calculate cumulative phase (phase) from delta phase (current_phase - last_phase)
        if self.frame_num == 0:
            logger.debug("First frame, using current phase as cumulative phase.")
            delta_phase = 0
            phase = current_phase
            self.last_phase = current_phase
        else:
            delta_phase = current_phase - self.last_phase
            while delta_phase < -np.pi:
                delta_phase += 2 * np.pi
            if self.frame_num < self.settings["frame_buffer_length"]:
                phase = self.frame_history[self.frame_num - 1, 1] + delta_phase
            else:
                phase = self.frame_history[-1, 1] + delta_phase
            self.last_phase = current_phase

        # Clears last entry of framerateSummaryHistory if it exceeds the reference frame length
        if self.frame_num >= self.settings["frame_buffer_length"]:
            self.frame_history = np.roll(self.frame_history, -1, axis=0)

        # Gets the argmin of SAD and adds to frame_history array
        #  TODO: JT writes: I don't know what this history is used for, but I think it's pretty weird to have a preallocated buffer,
        # and then just keep updating the final element if we overrun the buffer. I would have expected a FIFO buffer containing up to N recent frames...
        # That would also simplify subsequent logic e.g. "Predicting with partial buffer" would not be needed
        if self.frame_num < self.settings["frame_buffer_length"]:
            self.frame_history[self.frame_num, :] = (
                current_time,
                phase,
                np.argmin(sad),
            )
        else:
            self.frame_history[-1, :] = current_time, phase, np.argmin(sad)

        self.last_phase = float(current_phase)
        self.frame_num += 1

        logger.trace(self.frame_history[-1, :])
        logger.debug(
            "Current time: {0};, cumulative phase: {1} ({2:+f}); sad: {3}",
            current_time,
            phase,
            delta_phase,
            self.frame_history[-1, -1],
        )

        # If at least one period has passed, have a go at predicting a future trigger time
        if self.frame_num - 1 > self.pog_settings["referencePeriod"]:
            logger.debug("Predicting trigger...")

            # TODO: JT writes: this seems as good a place as any to highlight the general issue that the code is not doing a great job of precise timing.
            # It determines a delay time before sending the trigger, but then executes a bunch more code.
            # Oh and, more importantly, that delay time is then treated relative to “current_time”, which is set *after* doing the phase-matching.
            # That is going to reduce accuracy and precision, and also makes me even more uncomfortable in terms of future-proofing.
            # I think it would be much better to pass around absolute times, not deltas.

            # Gets the trigger response
            if self.frame_num < self.settings["frame_buffer_length"]:
                logger.trace("Predicting with partial buffer.")
                timeToWaitInSecs = pog.predict_trigger_wait(
                    self.frame_history[: self.frame_num :, :],
                    self.pog_settings,
                    fitBackToBarrier=True,
                    output="seconds",
                )
            else:
                logger.trace("Predicting with full buffer.")
                timeToWaitInSecs = pog.predict_trigger_wait(
                    self.frame_history,
                    self.pog_settings,
                    fitBackToBarrier=True,
                    output="seconds",
                )
            # frame_history is an nx3 array of [timestamp, phase, argmin(SAD)]
            # phase (i.e. frame_history[:,1]) should be cumulative 2Pi phase
            # targetSyncPhase should be in [0,2pi]

            # Captures the image  and then moves the stage if triggered
            if timeToWaitInSecs > 0:
                logger.info("Possible trigger after: {0}s", timeToWaitInSecs)

                (
                    timeToWaitInSecs,
                    sendTriggerNow,
                    self.pog_settings,
                ) = pog.decide_trigger(
                    current_time, timeToWaitInSecs, self.pog_settings
                )
                if sendTriggerNow != 0:
                    # TODO: JT writes: predict_trigger_wait uses variable timeToWaitInSecs, but this log labels it in ms. Which is it? [note that I have changed the variable name here to timeToWaitInSecs, but...]
                    logger.success(
                        "Sending trigger (reason: {0}) at time ({1} plus {2}) ms",
                        sendTriggerNow,
                        current_time,
                        timeToWaitInSecs,
                    )
                    # Trigger only
                    self.trigger_fluorescence_image_capture(
                        current_time + timeToWaitInSecs
                    )
                    if self.moveStagesAfterTrigger:
                        # Move stage
                        # TODO: JT writes: here again, why not pass the usb_stages settings, rather than passing these individual parameters and risking extra bugs?
                        stage_result = scf.move_stage(
                            self.usb_serial,
                            self.settings["usb_stages"]["plane_address"],
                            self.settings["usb_stages"]["increment"],
                            self.settings["usb_stages"]["encoding"],
                            self.settings["usb_stages"]["terminator"],
                        )
                        logger.info(stage_result)
                        # TODO: Do something with the stage result:
                        # 	0 = Continue as normal
                        # 	1 or 2 = Pause capture

                    # Store trigger time and update trigger number (for adaptive algorithm)
                    self.trigger_times.append(current_time + timeToWaitInSecs)
                    self.trigger_num += 1
                    # Returns the delay time, phase and timestamp (useful in the emulated scenario)
                    return timeToWaitInSecs, current_phase, current_time

        return None, current_phase, current_time

    def clear_state(self):
        """State 1 - re-initialise (clears for mode 2).
        Clears everything required to get a new period.debug
        Used if the user is not happy with a period choice
        Or before getting a new reference period in the adaptive mode.
        """
        logger.info("Resetting for new period determination.")
        self.frame_num = 0
        self.ref_frames = None
        self.ref_buffer = np.empty(
            (self.settings["frame_buffer_length"], self.width, self.height),
            dtype=self.dtype,
        )
        # TODO: JT writes: I don't like this logic - I don't feel this is the right place for it.
        # Also, update_after_n_triggers is one reason why we might want to reset the sync,
        # but the user should have the ability to reset the sync through the GUI, or there might
        # be other future reasons we might want to reset the sync (e.g. after each stack).
        # I think this could partly be tidied by making self.state behave more like a proper finite state machine,
        # A proper FSM would respond to the "reset" *input* slightly differently when in the "0" or "3" *state*.
        # A simpler tweak that is more consistent with the current design would just involve having two different states,
        # "reset for POG" and "reset for LTU". Both could call through to this same function, but then this function
        # knows what it should be doing.
        if (
            self.settings["update_after_n_triggers"] > 0
            and self.trigger_num >= self.settings["update_after_n_triggers"]
        ):
            # i.e. if adaptive reset trigger_num and get new period
            # automatically phase-locking with the existing period
            self.trigger_num = 0
            self.state = 3
        else:
            self.state = 2

    def get_period_state(self, frame):
        """ State 2 - get period mode (default behaviour requires user input).     
        In this mode we obtain a minimum number of frames,
        determine a period and then return.
        It is assumed that the user (or cli/flask app) then runs
        the select_period function (and updates the state)
        before running analyse again with the new state.
        """
        # TODO: JT writes: what user input? Why!? I think this function header needs updating - I'm not convinced any of what it says is accurate!
        # Or perhaps I am making too many assumptions, and this code is a bit less self-sufficient than I had imagined.
        # TODO: JT writes: I think this is a very odd way to do things (fill the ref_buffer before attempting to determine the period).
        # Why are you doing it like that? (the loop in establish_indices now makes more sense to me, at least...)
        # This is not going to work well live, when we want to lock on to a period ASAP, rather than acquiring an unnecessarily long frame buffer first.
        # (Is any of this due to concern about taking too much time to analyze one frame before the next one arrives...?)
        logger.debug("Processing frame in get period mode.")

        # Obtains a minimum amount of buffer frames
        if self.frame_num < self.settings["frame_buffer_length"]:
            logger.debug("Not yet enough frames to determine a new period.")

            # Adds current frame to buffer
            self.ref_buffer[self.frame_num, :, :] = frame

            # Increases frame number
            self.frame_num += 1

        # Once a suitable reference size has been buffered
        # gets a period and ask the user to select the target frame
        else:
            logger.info("Determining new reference period")
            # Calculate period from determine_reference_period.py
            self.ref_frames, self.pog_settings = ref.establish(
                self.ref_buffer, self.pog_settings
            )

            # Determine barrier frames
            self.pog_settings = pog.determine_barrier_frames(self.pog_settings)

            # Save the period
            ref.save_period(self.ref_frames, self.settings["period_dir"])
            logger.success("Period determined.")

            # Note, passing the new period to the adaptive system is left to the user/app
            self.stop = True

    def update_reference_sequence(self, frame):
        """ State 3 - adaptive mode (update reference sequence, while maintaining the same phase lock).
            In this mode we obtain a minimum number of frames
            Determine a period and then align with previous
            periods using an adaptive algorithm.
        """
        # TODO: JT writes: I really think this will need to be restructured so it processes "on the fly"
        # rather than accumulating a long sequence of frames and then analysing them all in one big chunk.
        # See my more extensive comments in separate document.
        logger.debug("Processing frame in update period mode.")

        # Obtains a minimum amount of buffer frames
        if self.frame_num < self.settings["frame_buffer_length"]:
            logger.debug("Not yet enough frames to determine a new period.")

            # Inserts current frame into buffer
            self.ref_buffer[self.frame_num, :, :] = frame

            # Increases frame number counter
            self.frame_num += 1

        # Once a suitable number of frames has been buffered,
        # gets a new period and aligns to the history
        else:
            # Obtains a reference period
            logger.debug("Determining new reference period")
            # Calculate period from determine_reference_period.py
            self.ref_frames, self.pog_settings = ref.establish(
                self.ref_buffer, self.pog_settings
            )

            # Determine barrier frames
            self.pog_settings = pog.determine_barrier_frames(self.pog_settings)

            # Save the period
            ref.save_period(self.ref_frames, self.settings["period_dir"])
            logger.success("Period determined.")

            self.state = 0

            self.frame_num = 0

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
                self.pog_settings["referencePeriod"],
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
            self.pog_settings = parameters.update(
                self.pog_settings,
                referenceFrame=(
                    self.pog_settings["referencePeriod"] * self.target / 80
                )  # TODO IS THIS CORRECT?    # TODO: JT writes: who wrote this comment? Happy to discuss, but would be nice to know who wrote this and why (and resolve it!)
                % self.pog_settings[
                    "referencePeriod"
                ],  # TODO: JT writes: what purpose does the modulo serve? I wouldn't have expected it to be needed... [makes me worry that there's a bug related to the extra frame padding!]
            )
            logger.success(
                "Reference period updated. New period of length {0} with reference frame at {1}",
                self.pog_settings["referencePeriod"],
                self.pog_settings["referenceFrame"],
            )

    def select_period(self, frame=None):
        """Selects the period from a set of reference frames

        Function inputs:
            self.ref_frames = a 3D array consisting of evenly spaced frames containing exactly one period
            self.pog_settings = the settings dictionary (for more information see the parameters.py file)

        Optional inputs:
            framerate = the framerate of the brightfield picam (float or int)
        """
        # Defines initial variables
        period_length_in_frames = self.ref_frames.shape[0]

        if frame is None:
            # For now it is a simple command line interface (which is not helpful at all)
            frame = int(
                input(
                    "Please select a frame between 0 and "
                    + str(period_length_in_frames - 1)
                    + "\nOr enter -1 to select a new period.\n"
                )
            )

        # Checks if user wants to select a new period. Users can use their creative side by selecting any negative number.
        if frame < 0:
            logger.success("User has asked for a new period to be determined.")
            self.state = 1
            return 1

        # Otherwise, if user is happy with period
        self.pog_settings = parameters.update(self.pog_settings, referenceFrame=frame)
        self.frame_num = 0
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
            self.pog_settings["referencePeriod"],
            self.pog_settings["drift"],
            max_offset=3,
            ref_seq_id=0,
            ref_seq_phase=frame,
        )

        # turn recording back on for rest of run
        self.stop = False

        return 0

    def trigger_fluorescence_image_capture(self, delay):
        """Triggers both the laser and fluorescence camera (assumes edge trigger mode by default) at a future time

        # Function inputs:
        # 		delay = delay time (in microseconds) before the image is captured
        # 		laser_trigger_pin = the pin number (int) of the laser trigger
        # 		fluorescence_camera_pins = an int array containg the triggering, SYNC-A and SYNC-B pin numbers for the fluorescence camera
        #
        # Optional inputs:    # TODO: JT writes: this is not an optional input, it is a class variable (which should really be a 'settings' element)
        # 		edge_trigger:
        # 			True = the fluorescence camera captures the image once detecting the start of an increased signal
        # 			False = the fluorescence camera captures for the duration of the signal pulse (pulse mode)
        # 		duration = (only applies to pulse mode [edge_trigger=False]) the duration (in microseconds) of the pulse
        # TODO: ABD add some logging here
        """

        # TODO: JT writes: needs better comment to explain what these modes are. What is "trigger mode" vs "edge mode"?
        # TODO: JT writes: really important to clarify that these calls to fp are *blocking* calls, i.e. they only return after the trigger has been sent
        # Captures an image in edge mode
        if self.edge_trigger:
            fp.edge(
                delay,
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"][0],
                self.settings["fluorescence_camera_pins"][2],
            )
        # Captures in trigger mode
        else:
            fp.pulse(
                delay,
                self.duration,
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"][0],
            )

    def plot_triggers(self, outfile="triggers.png"):
        """Plot the phase vs. time sawtooth line with trigger events."""
        self.timestamp = np.array(self.timestamp)
        self.phase = np.array(self.phase)
        self.trigger_times = np.array(self.trigger_times)

        plt.figure()
        plt.title("Zebrafish heart phase with trigger fires")
        plt.plot(self.timestamp, self.phase, label="Heart phase")
        plt.scatter(
            self.trigger_times,
            np.full(
                max(len(self.trigger_times), 0), self.pog_settings["targetSyncPhase"]
            ),
            color="r",
            label="Trigger fire",
        )
        # Add labels etc
        x_1, x_2, _, y_2 = plt.axis()
        plt.axis((x_1, x_2, 0, y_2 * 1.1))
        plt.legend()
        plt.xlabel("Time (ms)")
        plt.ylabel("Phase (rad)")

        # Saves the figure
        plt.savefig(outfile)
        plt.show()

    def plot_accuracy(self, outfile="accuracy.png"):
        """Plot the target phase and adjusted real phase of trigger events."""
        self.timestamp = np.array(self.timestamp)
        self.phase = np.array(self.phase)
        self.trigger_times = np.array(self.trigger_times)

        triggeredPhase = []
        for i in range(len(self.trigger_times)):

            triggeredPhase.append(
                self.phase[(np.abs(self.timestamp - self.trigger_times[i])).argmin()]
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

    def plot_running(self, outfile="running.png"):
        self.timestamps = np.array(self.timestamp)
        self.process_time = np.array(self.process_time)

        plt.figure()
        plt.title("Frame processing times")
        plt.plot(self.timestamp, self.process_time, label="Processing time")
        # Add labels etc
        plt.xlabel("Time (ms)")
        plt.ylabel("Processing time (ms)")

        # Saves the figure
        plt.savefig(outfile)
        plt.show()


def run(settings):
    """Run the software using a live PiCam."""

    # Initialise PiCamera
    logger.success("Initialising camera...")
    # Camera settings
    camera = picamera.PiCamera()
    camera.framerate = settings["brightfield_framerate"]
    camera.resolution = (
        settings["brightfield_resolution"],
        settings["brightfield_resolution"],
    )
    camera.awb_mode = settings["awb_mode"]
    camera.exposure_mode = settings["exposure_mode"]
    camera.shutter_speed = settings["shutter_speed"]  # us
    camera.image_denoise = settings["image_denoise"]

    # Initialise Gater
    logger.success("Initialising gater...")
    analyser = OpticalGater(camera=camera, settings=settings,)

    logger.success("Determining reference period...")
    while analyser.state > 0:
        camera.start_recording(analyser, format="yuv")
        while not analyser.stop:
            camera.wait_recording(0.001)  # s
        camera.stop_recording()
        logger.info("Requesting user input...")
        analyser.state = analyser.select_period(10)
    logger.success(
        "Period determined ({0} frames long) and user has selected frame {1} as target.",
        analyser.pog_settings["referencePeriod"],
        analyser.pog_settings["referenceFrame"],
    )

    logger.success("Running...")
    camera.start_recording(analyser, format="yuv")
    time_started = time.time()  # so we can time-out
    time_running = 0
    while not analyser.stop and time_running < settings["analyse_time"]:
        camera.wait_recording(0.001)  # s
        time_running = time.time() - time_started
    camera.stop_recording()

    logger.success("Plotting summaries...")
    analyser.plot_triggers()
    analyser.plot_accuracy()
    analyser.plot_running()

    logger.success("Fin.")


if __name__ == "__main__":
    # Reads data from settings json file
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = "settings.json"

    with open(settings_file) as data_file:
        settings = json.load(data_file)

    # Performs a live data capture
    run(settings)
