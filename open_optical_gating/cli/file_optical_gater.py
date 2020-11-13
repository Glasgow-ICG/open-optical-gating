"""Extension of CLI Open Optical Gating System for emulating gating with saved brightfield data"""

# Python imports
import sys
import json
import time

# Module imports
from loguru import logger
from skimage import io

# Local imports
from . import optical_gater_server as server
from . import pixelarray as pa


class FileOpticalGater(server.OpticalGater):
    """Extends the optical gater server for a pre-captured data file.
    """

    def __init__(
        self,
        source=None,
        settings=None,
        ref_frames=None,
        ref_frame_period=None,
        repeats=1,
        automatic_target_frame=True,
    ):
        """Function inputs:
            settings      dict  Parameters affecting operation (see default_settings.json)
        """

        # Get the assumed brightfield framerate from the settings
        self.framerate = settings["brightfield_framerate"]

        # Initialise parent
        super(FileOpticalGater, self).__init__(
            settings=settings, ref_frames=ref_frames, ref_frame_period=ref_frame_period,
        )

        # Load the data
        self.load_data(source)

        # How many times to repeat the sequence
        self.repeats_remaining = repeats

        # By default we will take a guess at a goof target frame (True)
        # rather than ask user for their preferred initial target frame (False)
        self.automatic_target_frame = automatic_target_frame

    def load_data(self, filename):
        """Load data file"""
        # Load
        logger.success("Loading image data...")
        self.data = io.imread(filename)
        self.height, self.width = self.data[0].shape

        # Initialise frame iterator and time tracker
        self.next_frame_index = 0
        self.start_time = time.time()  # we use this to sanitise our timestamps
        self.last_frame_wallclock_time = None

    def run_server(self, force_framerate=False):
        """ Run the OpticalGater server, acting on the in-file data.
            Function inputs:
                force_framerate bool    Whether or not to slow down the compute time to emulate real-world speeds
        """
        if self.automatic_target_frame == False:
            logger.success("Determining reference period...")
            while self.state != "sync":
                while not self.stop:
                    self.analyze_pixelarray(self.next_frame(force_framerate=True))
                logger.info("Requesting user input...")
                self.user_select_ref_frame()
            logger.success(
                "Period determined ({0} frames long) and user has selected frame {1} as target.",
                self.pog_settings["reference_period"],
                self.pog_settings["referenceFrame"],
            )

        logger.success("Emulating...")
        while not self.stop:
            self.analyze_pixelarray(self.next_frame(force_framerate=force_framerate))

    def next_frame(self, force_framerate=False):
        """This function gets the next frame from the data source, which can be passed to analyze()"""
        # Force framerate to match the brightfield_framerate in the settings
        # This gives accurate timings and plots
        if force_framerate and (self.last_frame_wallclock_time is not None):
            wait_s = (1 / self.settings["brightfield_framerate"]) - (
                time.time() - self.last_frame_wallclock_time
            )
            if wait_s > 1e-9:
                # the 1e-9 is a very small time to allow for the calculation
                time.sleep(wait_s - 1e-9)
            else:
                logger.warning(
                    "Failing to sustain requested framerate {0}fps for frame {1} (requested negative delay {2}s)".format(
                        self.settings["brightfield_framerate"],
                        self.next_frame_index,
                        wait_s,
                    )
                )

        if self.next_frame_index == self.data.shape[0] - 1:
            self.repeats_remaining -= 1
            if self.repeats_remaining <= 0:
                # If this is our last frame we set the stop flag for the user/app to know
                self.stop = True
            else:
                # Start again at the first frame in the file
                self.next_frame_index = 0

        next = pa.PixelArray(
            self.data[self.next_frame_index, :, :],
            metadata={
                "timestamp": time.time() - self.start_time
            },  # relative to start_time to sanitise
        )
        self.next_frame_index += 1
        self.last_frame_wallclock_time = time.time()
        return next


def run(settings):
    logger.success("Initialising gater...")
    analyser = FileOpticalGater(
        source=settings["path"], settings=settings, automatic_target_frame=False,
    )

    logger.success("Running server...")
    analyser.run_server(force_framerate=True)

    logger.success("Plotting summaries...")
    analyser.plot_triggers()
    analyser.plot_prediction()
    analyser.plot_accuracy()
    analyser.plot_running()


if __name__ == "__main__":
    # Reads data from settings json file
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = "settings.json"

    with open(settings_file) as data_file:
        settings = json.load(data_file)

    # Runs the server
    run(settings)
