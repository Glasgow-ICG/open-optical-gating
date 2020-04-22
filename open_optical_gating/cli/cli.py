"""Main CLI Open Optical Gating System"""

import os
import time
from datetime import datetime
from tqdm import tqdm
from loguru import logger

# Numpy, Scipy, Pandas and Scikits
import numpy as np
import pandas as pd
from skimage import io

# Matplotlib
import matplotlib.pyplot as plt

# PiCamera imports
import picamera
from picamera import array  # TODO: can I remove this and use picamera.array?

# Serial imports - for stages? # TODO: move stage stuff into separate file?
import serial

import json  # TODO move this into parameters.py

# Fastpins
import fastpins as fp

# Optical Gating Alignment
import optical_gating_alignment.optical_gating_alignment as oga

# Local imports
import open_optical_gating.cli.determine_reference_period as ref
import open_optical_gating.cli.prospective_optical_gating as pog
import open_optical_gating.cli.parameters as parameters
import open_optical_gating.cli.stage_control_functions as scf

import sys

logger.remove()
logger.add(sys.stderr, level="SUCCESS")
logger.enable("open_optical_gating")


class YUVLumaAnalysis(array.PiYUVAnalysis):
    """Custom class to convert and analyse Y (luma) channel of each YUV frame.
    Extends the picamera.array.PiYUVAnalysis class, which has a stub method called analze that is overidden here.
    """

    def __init__(
        self,
        camera=None,
        usb_serial=None,
        brightfield_framerate=80,
        laser_trigger_pin=22,
        fluorescence_camera_pins=(8, 10, 12),
        plane_address=1,
        encoding="utf-8",
        terminator=chr(13) + chr(10),
        increment=0.0005,
        negative_limit=0,
        positive_limit=0.075,
        current_position=0,
        frame_buffer_length=100,
        ref_frames=None,
        frame_num=0,
        live=True,
        output_mode="glaSPIM",
        updateAfterNTriggers=200,
        duration=1e3,
        period_dir="~/",
    ):
        # Function inputs:
        # 	camera = the raspberry picam PiCamera object
        # 	laser_trigger_pin = the pin number (int) of the laser trigger
        # 	fluorescence_camera_pins = an array (int) of fluorescence camera pin numbers containg (trigger,SYNC-A, SYNC-B)
        # 	usb_serial = the usb serial object for controlling the movement stages
        # 	plane_address = the address of the stage that moves the zebrafish through the light sheet
        # 	encoding = the encoding used to control the Newport stages (usually utf-8)
        # 	terminator = the character set used to terminate a command sent to the Newport stages
        # 	increment = the required increment to move the stage by after each image capture (float)
        # 	negative_limit = the smallest z value (float) of the edge of the zebrafish heart (selected by the user)
        # 	positive_limit = the largest z value (float) of the edge of the zebrafish heart (selected by the user)
        # 	current_position = the current z value of the stage.

        # Optional inputs:
        # 	ref_frames = a set of reference frames containg a whole period for the zebrafish
        # 	frame_num = the current frame number

        super(YUVLumaAnalysis, self).__init__(camera)
        self.frame_num = frame_num
        if camera is not None:
            self.width, self.height = camera.resolution
        else:
            self.width, self.height = (115, 170)  # for emulated sequence

        self.framerate = brightfield_framerate

        # Defines laser, fluorescence camera and usb serial information
        self.laser_trigger_pin = laser_trigger_pin
        self.fluorescence_camera_pins = fluorescence_camera_pins

        self.usb_serial = usb_serial
        self.plane_address = plane_address
        self.encoding = encoding
        self.terminator = terminator
        self.increment = increment
        self.negative_limit = negative_limit
        self.positive_limit = positive_limit
        self.current_position = current_position

        self.camera = camera

        # Defines the arrays for sad and frameSummaryHistory (which contains period, timestamp and argmin(sad))
        self.frame_buffer_length = frame_buffer_length
        self.frameSummaryHistory = np.zeros((self.frame_buffer_length, 3))
        self.dtype = "uint8"
        self.ref_frames = ref_frames

        # Variables for adaptive algorithm
        # Should be compatible with standard adaptive algorithm
        #  (Nat Comms paper) and dynamic algorithm (unpublished, yet)
        self.updateAfterNTriggers = updateAfterNTriggers  # if 0 turns adaptive mode off
        self.trigger_num = 0
        self.resampledSequence = []
        self.periodHistory = []
        self.shifts = []
        self.driftHistory = []

        # Variable for emulator
        self.live = live
        self.targetSyncPhaseOld = -1

        # Array for fps test
        self.time_ary = []

        # Start experiment timer
        self.initial_process_time = time.time()

        # Sets ouput mode, reverts to Glasgow SPIM mode by default (mode specified in JSON file)
        if output_mode == "5V_BNC_Only":
            self.outputMode = 1
        else:
            self.outputMode = 0

        # Initialises reference frames if not specified
        # get_period has several modes:
        #   0 - run prospective gating mode (phase locked triggering)
        #   1 - re-initialise (clears for mode 2)
        #   2 - get period mode (requires user input)
        #   3 - adaptive mode (update period but maintain phase lock)
        if self.ref_frames is None:
            logger.info("No reference frames found, switching to get period mode.")
            self.ref_buffer = np.empty(
                (self.frame_buffer_length, self.height, self.width), dtype=self.dtype
            )
            self.state = 2

        else:
            logger.debug("Using existing reference frames.")
            self.state = 0
            self.settings = parameters.initialise(
                framerate=self.framerate, referencePeriod=ref_frames.shape[0]
            )
            self.settings = pog.determine_barrier_frames(self.settings)
            self.initial_process_time = time.time()

        self.period_dir = period_dir

        self.progress = 0  # for webapp progress bars

        # DEVNOTE - if the laser doesn't trigger check this line works
        # trigger_fluorescence_image_capture(0, laser_trigger_pin, fluorescence_camera_pins, edge_trigger=False, duration=duration)

    def analyze(self, frame):
        """ Method to analyse each frame as they are captured by the camera.
        Must be fast since it is running within the encoder's callback,
        and so must return before the next frame is produced.
        Essentialy this method just calls the appropriate method based on the state attribute."""

        # For logging processing time
        time_init = time.time()

        # if we're passed a colour image, take the first channel
        if len(frame.shape) == 3:
            frame = frame[:, :, 0]

        if self.trigger_num >= self.updateAfterNTriggers:
            # time to update the reference period whilst maintaining phase lock
            # set state to 1 (so we clear things for a new reference period)
            # as part of this trigger_num will be reset
            self.state = 1

        # Ensures stage is always within user defined limits (only needed for RPi-controlled stages)
        stages_safe = self.current_position <= self.positive_limit and self.current_position >= self.negative_limit
        if stages_safe:
            if self.state == 0:
                # Using determined reference peiod, analyse brightfield frames
                # to determine prospective optical gating triggers
                (trigger_response, pp, tt) = self.pog_state(frame)

                # Logs processing time
                time_fin = time.time()
                (self.time_ary).append(time_fin - time_init)

                return trigger_response, pp, tt

            elif self.state == 1:
                # Clears reference period and resets frame number
                # Used when determining new period
                self.clear_state()

            elif self.state == 2:
                # Determine initial reference period and target frame
                self.get_period_state(frame)

                # Logs processing time
                time_fin = time.time()
                (self.time_ary).append(time_fin - time_init)

                if self.ref_frames is not None:
                    logger.debug('Returning reference frames for target fram selection.')
                    # Return the determined period
                    # The user (or app) then needs to run self.select_period()
                    # (updating self.state) before starting the analyse again
                    return (
                        None,
                        -1,
                        -1,
                    )  # a negative phase and time can be caught by the parent program

            elif self.state == 3:
                # Determine reference period syncing target frame with original user selection
                self.update_period_state(frame)

            else:
                logger.critical("Unknown state {0}.", self.state)

        # Logs processing time
        time_fin = time.time()
        (self.time_ary).append(time_fin - time_init)

        return None, None, None  # default return of response, pp, tt all empty

    def pog_state(self, frame):
        """State 0 - run prospective optical gating mode (phase locked triggering)."""
        logger.debug("Processing frame in prospective optical gating mode.")

        # Gets the phase (in frames) and sad of the current frame
        pp, sad, self.settings = pog.phase_matching(
            frame, self.ref_frames, settings=self.settings
        )
        logger.trace(sad)

        # Convert phase to 2pi base
        pp = (
            2
            * np.pi
            * (pp - self.settings["numExtraRefFrames"])
            / self.settings["referencePeriod"]
        )  # rad

        # Gets the current timestamp in milliseconds
        tt = (
            time.time() - self.initial_process_time
        ) * 1000  # Converts time into milliseconds

        # Calculate cumulative phase (phase) from delta phase (pp - pp_old)
        if self.frame_num == 0:
            logger.debug('First frame, using pp as cumulative phase.')
            deltaPhase = 0
            phase = pp
            self.pp_old = pp
        else:
            deltaPhase = pp - self.pp_old
            while deltaPhase < -np.pi:
                deltaPhase += 2 * np.pi
            if self.frame_num < self.frame_buffer_length:
                phase = self.frameSummaryHistory[self.frame_num - 1, 1] + deltaPhase
            else:
                phase = self.frameSummaryHistory[-1, 1] + deltaPhase
            self.pp_old = pp

        # Clears last entry of framerateSummaryHistory if it exceeds the reference frame length
        if self.frame_num >= self.frame_buffer_length:
            self.frameSummaryHistory = np.roll(self.frameSummaryHistory, -1, axis=0)

        # Gets the argmin of SAD and adds to frameSummaryHistory array
        if self.frame_num < self.frame_buffer_length:
            self.frameSummaryHistory[self.frame_num, :] = (
                tt,
                phase,
                np.argmin(sad),
            )
        else:
            self.frameSummaryHistory[-1, :] = tt, phase, np.argmin(sad)

        self.pp_old = float(pp)
        self.frame_num += 1

        logger.trace(self.frameSummaryHistory[-1,:])
        logger.debug(
            "Current time: {0};, cumulative phase: {1} ({2:+f}); sad: {3}",
            tt,
            phase,
            deltaPhase,
            self.frameSummaryHistory[-1, -1],
        )

        # If at least one period has passed, have a go at predicting a trigger
        if self.frame_num - 1 > self.settings["referencePeriod"]:
            logger.debug("Predicting trigger...")

            # Gets the trigger response
            if self.frame_num < self.frame_buffer_length:
                logger.trace('Triggering with partial buffer.')
                trigger_response = pog.predict_trigger_wait(
                    self.frameSummaryHistory[:self.frame_num:, :],
                    self.settings,
                    fitBackToBarrier=True,
                    output="seconds",
                )
            else:
                logger.trace('Triggering with full buffer.')
                trigger_response = pog.predict_trigger_wait(
                    self.frameSummaryHistory,
                    self.settings,
                    fitBackToBarrier=True,
                    output="seconds",
                )
            # frameSummaryHistory is an nx3 array of [timestamp, phase, argmin(SAD)]
            # phase (i.e. frameSummaryHistory[:,1]) should be cumulative 2Pi phase
            # targetSyncPhase should be in [0,2pi]

            # Captures the image  and then moves the stage if triggered
            if trigger_response > 0:
                logger.info("Possible trigger: {0}", trigger_response)

                (trigger_response, send, self.settings,) = pog.decide_trigger(
                    tt, trigger_response, self.settings
                )
                if send > 0:

                    self.trigger_num = (
                        self.trigger_num + 1
                    )  # update trigger number for adaptive algorithm
                    logger.success("Sending trigger: {0}", send)
                    if self.live:

                        if self.outputMode == 1:
                            trigger_fluorescence_image_capture(
                                tt + trigger_response,
                                self.laser_trigger_pin,
                                self.fluorescence_camera_pins,
                                edge_trigger=False,
                                duration=1e3,
                            )
                        else:

                            trigger_fluorescence_image_capture(
                                tt + trigger_response,
                                self.laser_trigger_pin,
                                self.fluorescence_camera_pins,
                                edge_trigger=False,
                                duration=1e3,
                            )
                            stage_result = scf.move_stage(
                                self.usb_serial,
                                self.plane_address,
                                self.increment,
                                self.encoding,
                                self.terminator,
                            )
                            logger.info(stage_result)
                    else:
                        logger.info(
                            "Not Live: {0}, {1}, {2}, {3}",
                            send,
                            trigger_response,
                            pp,
                            tt,
                        )
                        # Returns the trigger response, phase and timestamp for emulated data
                        return trigger_response, pp, tt

                    # self.targetSyncPhaseOld = current_sync_phase
                elif not self.live:
                    logger.info(
                        "Not sent: {0}, {1}, {2}, {3}", send, trigger_response, pp, tt,
                    )
                    return None, pp, tt

                # Do something with the stage result:
                # 	0 = Continue as normal
                # 	1 or 2 = Pause capture

        return None, pp, tt

    def clear_state(self):
        """State 1 - re-initialise (clears for mode 2).
        Clears everything required to get a new period.
        Used if the user is not happy with a period choice
        Or before getting a new reference period in the adaptive mode.
        """
        logger.info("Resetting for new period determination.")
        self.frame_num = 0
        self.ref_frames = np.zeros(
            (self.frame_buffer_length, self.height, self.width), dtype=self.dtype,
        )
        if (
            self.updateAfterNTriggers > 0
            and self.trigger_num >= self.updateAfterNTriggers
        ):
            # i.e. if adaptive reset trigger_num and get new period
            # automatically phase-locking with the existing period
            self.trigger_num = 0
            self.state = 3
        else:
            self.state = 2

    def get_period_state(self, frame):
        """ State 2 - get period mode (default requires user input).
        In this mode we obtain a minimum number of frames
        Determine a period and then return.
        It is assumed that the user (or cli/flask app) then runs
        the select_period function (and updates the state)
        before running analyse again with the new state.
        """
        logger.debug("Processing frame in get period mode.")

        # Obtains a minimum amount of buffer frames
        if self.frame_num < self.frame_buffer_length:
            logger.debug("Not yet enough frames to determine a new period.")

            # Adds current frame to buffer
            self.ref_buffer[self.frame_num, :, :] = frame

            # Increases frame number
            self.frame_num += 1

        # Once a suitible reference size has been buffered
        # gets a period and ask the user to select the target frame
        else:
            logger.debug("Determining new reference period")

            # Obtains a reference period
            self.ref_frames, self.settings = get_period(
                self.ref_buffer,
                {},
                framerate=self.framerate,
                period_dir=self.period_dir,
            )
            logger.success("Period determined.")

    def update_period_state(self, frame):
        """State 3 - adaptive mode (update period but maintain phase lock).
        In this mode we obtain a minimum number of frames
        Determine a period and then align with previous
        periods using an adaptive algorithm.
        """
        logger.debug("Processing frame in update period mode.")

        # Obtains a minimum amount of buffer frames
        if self.frame_num < self.frame_buffer_length:
            logger.debug("Not yet enough frames to determine a new period.")

            # Adds current frame to buffer
            self.ref_buffer[self.frame_num, :, :] = frame

            # Increases frame number
            self.frame_num += 1

        # Once a suitable number of frames has been buffered
        # gets a period and align to the history
        else:
            logger.debug("Determining new reference period")

            # Obtains a reference period
            self.ref_frames, self.settings = get_period(
                self.ref_buffer,
                {},
                framerate=self.framerate,
                period_dir=self.period_dir,
            )
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
                self.settings["referencePeriod"],
                self.settings["drift"],
                sequence_history=self.sequence_history,
                period_history=self.period_history,
                drift_history=self.drift_history,
                shift_history=self.shift_history,
                global_solution=self.global_solution,
                max_offset=3,
                ref_seq_id=0,
                ref_seq_phase=self.settings["referenceFrame"],
            )
            self.settings = parameters.update(
                self.settings,
                referenceFrame=(
                    self.settings["referencePeriod"] * self.target / 80
                )  # TODO IS THIS CORRECT?
                % self.settings["referencePeriod"],
            )
            logger.success(
                "Reference period updated. New period of length {0} with reference frame at {1}",
                self.settings["referencePeriod"],
                self.settings["referenceFrame"],
            )

    def emulate(self):
        """ Function to emulate live data based on a saved multi-page TIFF."""

        # Defines initial variables and objects
        timestamp = []
        phase = []
        process_time = []
        trigger_times = []
        self.live = False
        self.targetSyncPhaseOld = -1

        number_of_frames = len(self.emulated_data) - self.emulate_start
        logger.trace(self.emulated_data.shape)

        for i, frame in enumerate(self.emulated_data[self.emulate_start : :]):
            logger.debug("Processing frame {0} of {1}", i, number_of_frames)
            self.progress = 100 * i / number_of_frames  # for static progress bar
            # yield "data:{0:.0f}\n\n".format(100*i/number_of_frames)  # for dynamic progress bar

            # Only send to analyze() if an initial period has been determined and a target frame set
            # i.e. all states but 2 (get_period_state())
            if self.state != 2:
                # Gets data from analyse function (also times function call)
                time_init = time.time()
                trigger_response, pp, tt = self.analyze(frame)
                logger.trace("t = {0}; p = {1}; tr = {2};", tt, pp, trigger_response)
                time_fin = time.time()

                # Adds data to lists
                if tt is None:
                    continue
                if tt != 0:
                    process_time.append(time_fin - time_init)
                    timestamp.append(tt)
                    phase.append(pp)
                    # If sucessful trigger response and sync phase has increased
                    if (
                        trigger_response is not None and trigger_response
                    ):  # != 0 and current_sync_phase > self.targetSyncPhaseOld:
                        trigger_times.append(trigger_response + tt)
                        # self.targetSyncPhaseOld = current_sync_phase
                else:
                    logger.warning("This is wrong!")

            # Gets period if trigger conditions are not met
            else:
                logger.critical("No period has been determined! (State {0})", self.state)
                return None

        # Converts lists to numpy arrays
        process_time = np.array(process_time)
        timestamp = np.array(timestamp)
        phase = np.array(phase)
        trigger_times = np.array(trigger_times)

        logger.info(
            "Processing time (min and max): {0} {1}",
            process_time.min(),
            process_time.max(),
        )
        logger.info(
            "Timestamp (min and max): {0} {1}", timestamp.min(), timestamp.max()
        )
        logger.info("Phase (min and max): {0} {1}", phase.min(), phase.max())
        logger.success(trigger_times)

        # Should have a sawtooth for Phase vs time and scatter points should lie on the saw tooth
        plt.figure()
        plt.title("Zebrafish heart phase with simulated trigger fire")
        plt.plot(timestamp, phase, label="Heart phase")
        plt.scatter(
            trigger_times[0:-1],
            np.full(max(len(trigger_times) - 1, 0), self.settings["targetSyncPhase"]),
            color="r",
            label="Simulated trigger fire",
        )
        # Add labels etc
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, y2 * 1.1))
        plt.legend()
        plt.xlabel("Time (ms)")
        plt.ylabel("Phase (rad)")

        # Saves the figure
        plt.savefig(
            os.path.join(
                "open_optical_gating", "app", "static", "triggers.png"
            ),
            dpi=1000,
        )
        plt.show()

        triggeredPhase = []
        for i in range(len(trigger_times)):

            triggeredPhase.append(
                phase[(np.abs(timestamp - trigger_times[i])).argmin()]
            )

        plt.figure()
        plt.title("Frequency density of triggered phase")
        bins = np.arange(0, 2 * np.pi, 0.1)
        plt.hist(triggeredPhase, bins=bins, color="g", label="Triggered phase")
        x1, x2, y1, y2 = plt.axis()
        plt.plot(
            np.full(2, self.settings["targetSyncPhase"]),
            (y1, y2),
            "r-",
            label="Target phase",
        )
        plt.xlabel("Triggered phase (rad)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.axis((x1, x2, y1, y2))

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "open_optical_gating", "app", "static", "accuracy.png"
            ),
            dpi=1000,
        )
        plt.show()

    def emulate_get_period(self, video_file):
        """ Function to emulate getting reference period"""

        # Defines initial variables and objects
        self.emulated_data = io.imread(video_file)
        self.live = False
        self.targetSyncPhaseOld = -1

        pp = 0
        i = 0
        while pp != -1:
            # Reads a frame from the emulated data set
            frame = self.emulated_data[i, :, :]
            _, pp, _ = self.analyze(frame)
            i = i + 1
        self.emulate_start = i

    def select_period(self, frame=None):
        """Selects the period from a set of reference frames

        Function inputs:
            self.ref_frames = a 3D array consisting of evenly spaced frames containing exactly one period
            self.settings = the settings dictionary (for more information see the helper.py file)

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
            return 1

        # Otherwise, if user is happy with period
        self.settings = parameters.update(self.settings, referenceFrame=frame)
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
            self.settings["referencePeriod"],
            self.settings["drift"],
            max_offset=3,
            ref_seq_id=0,
            ref_seq_phase=frame,
        )
        self.initialTarget = frame

        return 0


def init_controls(settings):
    """Function that initialises various controlls (pins for triggering laser and fluorescence camera along with the USB for controlling the Newport stages)
    Function inputs:
        laser_trigger_pin = the GPIO pin number connected to fire the laser
        fluorescence_camera_pins = an array of 3 pins used to for the fluoresence camera
                                        (trigger, SYNC-A, SYNC-B)
        usb_information = a list containing the information used to set up the usb for controlling the Newport stages
                                (USB address (str),timeout (flt), baud rate (int), byte size (int), parity (char), stop bits (int), xonxoff (bool))
    """

    # Defines initial variables
    laser_trigger_pin = settings["laser_trigger_pin"]
    fluorescence_camera_pins = settings[
        "fluorescence_camera_pins"
    ]  # Trigger, SYNC-A, SYNC-B
    usb_information = settings["usb_stages"]
    # will be either None (no stages) or a dict of:
    # 'usb_name', 'usb_timeout', 'usb_baudrate', 'usb_dataBits', 'usb_parity', 'usb_XOnOff'
    # usb_information needs to be a tuple of
    # (USB address, timeout, baud rate, data bits, parity, Xon/Xoff, True)
    if usb_information is not None:
        usb_information = (
            usb_information["usb_name"],
            usb_information["usb_timeout"],
            usb_information["usb_baudrate"],
            usb_information["usb_dataBits"],
            usb_information["usb_parity"],
            usb_information["usb_XOnOff"],
            True,
        )
        # TODO: ABD to make rest of code work with dict instead of tuple
        # TODO: ABD to work out what True is for!

    # Initialises fastpins module
    try:
        fp.init()
    except Exception as inst:
        logger.critical("Error setting up fastpins module.")
        logger.critical(inst)
        return 1

    # Sets up laser trigger pin
    if laser_trigger_pin is not None:
        try:
            fp.setpin(
                laser_trigger_pin, 1, 0
            )  # PUD resistor needs to be specified but will be ignored in setup
        except Exception as inst:
            logger.critical("Error setting up laser pin.")
            logger.critical(inst)
            return 2

    # Sets up fluorescence camera pins
    if fluorescence_camera_pins is not None:
        try:
            fp.setpin(fluorescence_camera_pins[0], 1, 0)  # Trigger
            fp.setpin(fluorescence_camera_pins[1], 0, 0)  # SYNC-A
            fp.setpin(fluorescence_camera_pins[2], 0, 0)  # SYNC-B
        except Exception as inst:
            logger.critical("Error setting up fluorescence camera pins.")
            logger.critical(inst)
            return 3

    # Sets up USB for Newport stages
    if usb_information is not None:
        try:
            ser = scf.init_stage(usb_information)

            # Serial object is the only new object
            return ser
        except Exception as inst:
            logger.critical("Error setting up usb.")
            logger.critical(inst)
            return 4

    return 0  # default return if no stage


def trigger_fluorescence_image_capture(
    delay, laser_trigger_pin, fluorescence_camera_pins, edge_trigger=True, duration=1e3
):
    """Triggers both the laser and fluorescence camera (assumes edge trigger mode by default)

    # Function inputs:
    # 		delay = delay time (in microseconds) before the image is captured
    # 		laser_trigger_pin = the pin number (int) of the laser trigger
    # 		fluorescence_camera_pins = an int array containg the triggering, SYNC-A and SYNC-B pin numbers for the fluorescence camera
    #
    # Optional inputs:
    # 		edge_trigger:
    # 			True = the fluorescence camera captures the image once detecting the start of an increased signal
    # 			False = the fluorescence camera captures for the duration of the signal pulse (pulse mode)
    # 		duration = (only applies to pulse mode [edge_trigger=False]) the duration (in microseconds) of the pulse
    # TODO: ABD add some logging here
    """

    # Captures an image in edge mode
    if edge_trigger:

        fp.edge(
            delay,
            laser_trigger_pin,
            fluorescence_camera_pins[0],
            fluorescence_camera_pins[2],
        )

    # Captures in trigger mode
    else:

        fp.pulse(delay, duration, laser_trigger_pin, fluorescence_camera_pins[0])


# Gets the period from sample set
def get_period(
    brightfield_sequence,
    settings,
    framerate=80,
    minFramesForFit=5,
    maxRecievedFramesForFit=80,
    predictionLatency=15,
    period_dir="~/",
):

    # Function inputs
    # 		brightfield_sequence = (numpy array) a 3D array of the brightfiled picam data
    # 		settings = the settings

    # If the settings are empty creates settings
    if not settings:
        settings = parameters.initialise(
            framerate=framerate,
            referencePeriod=brightfield_sequence.shape[0],
            minFramesForFit=minFramesForFit,
            predictionLatency=predictionLatency,
        )

    # Calculates period from determine_reference_period.py
    brightfield_period, settings = ref.establish(brightfield_sequence, settings)
    settings = pog.determine_barrier_frames(settings)

    # Add new folder with time stamp
    dt = datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(os.path.join(period_dir, dt), exist_ok=True)

    # Saves the period
    if isinstance(brightfield_period, int) == False:

        for i in range(brightfield_period.shape[0]):
            io.imsave(
                os.path.join(period_dir, dt, "{0:03d}.tiff".format(i)),
                brightfield_period[i, :, :],
            )

    return brightfield_period, settings


#  TODO: move emulate to a separate script? Maybe an examples module/script?
# Defines the three main modes (emulate capture, check fps and live data capture)
def emulate_data_capture(dict_data):
    # Emulated data capture for a set of sample data
    # emulate_data_set = "sample_data.tif"
    # emulate_data_set = 'sample_data.h264'
    logger.success("Initialising emulation...")
    analyse_camera = YUVLumaAnalysis(
        output_mode=dict_data["output_mode"],
        updateAfterNTriggers=dict_data["updateAfterNTriggers"],
        period_dir=dict_data["period_dir"],
    )
    logger.success("Determining reference period...")
    analyse_camera.emulate_get_period(dict_data["path"])
    logger.success("Getting user-input...")
    analyse_camera.state = analyse_camera.select_period(10)
    # deal with being asked to get new reference period
    while analyse_camera.state > 0:
        analyse_camera.emulate_get_period(dict_data["path"])
        analyse_camera.state = analyse_camera.select_period(10)
    logger.success("Emulating...")
    analyse_camera.emulate()
    logger.success("Fin.")


# TODO: is this needed/can this be incorporated elsewhere?
# Checks that the analyze function can run at the desired framerate
def check_fps(brightfield_framerate=80, brightfield_resolution=128):

    # Defines initial variables
    analyse_time = 10

    # Sets up basic picam
    camera = picamera.PiCamera()
    camera.framerate = brightfield_framerate
    camera.resolution = (brightfield_resolution, brightfield_resolution)

    # Generate fake reference frame set
    dummy_reference_frames = np.random.randint(
        0,
        high=128,
        size=(10, brightfield_resolution, brightfield_resolution),
        dtype=np.uint8,
    )

    # Sets up YUVLumaAnalysis object
    analyse_camera = YUVLumaAnalysis(
        camera=camera,
        brightfield_framerate=brightfield_framerate,
        ref_frames=dummy_reference_frames,
        live=False,
    )

    # Starts analysing brightfield data
    camera.start_recording(analyse_camera, format="yuv")
    camera.wait_recording(analyse_time)
    camera.stop_recording()

    # Gets longest time of analyze function
    longest_analyse_time = max(analyse_camera.time_ary)
    logger.debug("Longest call to 'analyze()'", longest_analyse_time)

    if 1 / longest_analyse_time > brightfield_framerate:

        logger.success("Success at fps: " + str(brightfield_framerate))
        camera.close()
        return brightfield_framerate, brightfield_resolution

    else:
        logger.success("Unsucessful at fps: " + str(brightfield_framerate))
        brightfield_framerate += -10
        camera.close()
        check_fps(brightfield_framerate=brightfield_framerate)


# Performs a live capture of the data
def live_data_capture(dict_data):
    # Initialise signallers
    # usb_serial will be one of:
    # 0 - if no failure and no usb stage
    # 1 - if fastpins fails
    # 2 - if laser pin fails
    # 3 - if camera pins fail
    # 4 - if usb stages fail
    # serial object - if no failure and usb stages desired
    usb_serial = init_controls(dict_data)
    usb_information = dict_data["usb_stages"]

    # Checks if usb_serial has recieved an error code
    if isinstance(usb_serial, int) and usb_serial > 0:
        ## TODO: replace with a true exception
        logger.critical("Error code {0}", usb_serial)
        return False
    elif isinstance(usb_serial, int) and usb_serial == 0:
        usb_serial = None
    else:
        # Defines variables for USB serial stage commands
        plane_address = usb_information["plane_address"]
        encoding = usb_information["encoding"]
        terminator = chr(usb_information["terminators"][0]) + chr(
            usb_information["terminators"][1]
        )

        # Sets up stage to recieve input
        neg_limit, pos_limit, current_position = scf.set_user_stage_limits(
            usb_serial, plane_address, encoding, terminator
        )

    # Camera settings
    camera = picamera.PiCamera()
    camera.framerate = dict_data["brightfield_framerate"]
    camera.resolution = (
        dict_data["brightfield_resolution"],
        dict_data["brightfield_resolution"],
    )
    camera.awb_mode = dict_data["awb_mode"]
    camera.exposure_mode = dict_data["exposure_mode"]
    camera.shutter_speed = dict_data["shutter_speed"]  # us
    camera.image_denoise = dict_data["image_denoise"]

    # Sets up YUVLumaAnalysis object
    # DEVNOTE: Remember to update the equivalent line in emulate_data_capture()
    if usb_serial is not None:
        analyse_camera = YUVLumaAnalysis(
            camera=camera,
            usb_serial=usb_serial,
            output_mode=dict_data["output_mode"],
            updateAfterNTriggers=dict_data["updateAfterNTriggers"],
            duration=dict_data["fluorescence_exposure"],
            period_dir=dict_data["period_dir"],
            brightfield_framerate=camera.framerate,
            increment=usb_information["increment"],
            negative_limit=neg_limit,
            positive_limit=pos_limit,
            current_position=current_position,
            plane_address=plane_address,
            encoding=encoding,
            terminator=terminator,
            # laser_trigger_pin=22,  # TODO
            # fluorescence_camera_pins=(8, 10, 12),  # TODO
            # frame_buffer_length=100,  # TODO
            # ref_frames=None,  # TODO
            # frame_num=0,  # TODO
        )
    else:
        analyse_camera = YUVLumaAnalysis(
            camera=camera,
            usb_serial=usb_serial,
            brightfield_framerate=camera.framerate,
            output_mode=dict_data["output_mode"],
            updateAfterNTriggers=dict_data["updateAfterNTriggers"],
            duration=dict_data["fluorescence_exposure"],
            period_dir=dict_data["period_dir"],
        )

    # Starts analysing brightfield data
    camera.start_recording(analyse_camera, format="yuv")
    camera.wait_recording(dict_data["analyse_time"])  # s
    camera.stop_recording()


if __name__ == "__main__":
    import sys

    # Iterates through a sample stack (with no period)
    # 	neg_limit = 0
    # 	pos_limit = 4
    # 	current_position = 0
    # 	increment = 0.01
    # 	plane_address = 1
    # 	encoding = 'utf-8'
    # 	terminator = chr(13)+chr(10)
    # 	delay = 400000
    # 	duration = 20000
    #
    # 	for i in tqdm(range(neg_limit, int(pos_limit/increment))):
    #
    # 		trigger_fluorescence_image_capture(delay, laser_trigger_pin, fluorescence_camera_pins, edge_trigger=False, duration=duration)
    # 		stage_result = scf.move_stage(usb_serial, plane_address,increment, encoding, terminator)
    # 	scf.move_stage(usb_serial, plane_address, pos_limit*(-1), encoding, terminator)

    # Reads data from json file
    if len(sys.argv) > 1:
        settings = sys.argv[1]
    else:
        settings = "settings.json"

    with open(settings) as data_file:
        dict_data = json.load(data_file)

    # Performs a live or emulated data capture
    live_capture = dict_data["live"]
    if live_capture == True:
        live_data_capture(dict_data)
    else:
        emulate_data_capture(dict_data)
