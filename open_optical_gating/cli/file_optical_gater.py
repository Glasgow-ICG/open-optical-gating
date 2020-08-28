"""Extension of CLI Open Optical Gating System for Emulation"""

# Python imports
import sys
import json
import time

# Module imports
from loguru import logger
from skimage import io

# Local imports
import open_optical_gating.cli.pixelarray as pa
import open_optical_gating.cli.optical_gater_server as server


class FileOpticalGater(server.OpticalGater):
    """Extends the optical gater server for a data file.
    """

    def __init__(
        self,
        file_source_path=None,
        settings=None,
        ref_frames=None,
        ref_frame_period=None,
        repeats=1
    ):
        """Function inputs:
            file_source_path   str   Path to file we will read as our input
            settings           dict  Parameters affecting operation (see default_settings.json)
            repeats            int   Number of times to repeat the frames in the source file.
                                     It may be useful to run through the frames more than once,
                                     for more sustained testing.
        """

        # initialise parent
        super(FileOpticalGater, self).__init__(
            settings=settings, ref_frames=ref_frames, ref_frame_period=ref_frame_period,
        )
        self.load_data(file_source_path)
        self.next_frame_index = 0
        self.last_frame_wallclock_time = None
        self.repeats_remaining = repeats
        # By default we will ask user for their preferred initial target frame
        self.automatic_target_frame = False

    def load_data(self, filename):
        """Apply data source-related settings."""
        # Data-source settings
        logger.success("Loading image data...")
        self.data = io.imread(filename)
        self.height, self.width = self.data[0].shape
        self.framerate = self.settings["brightfield_framerate"]
        self.start_time = time.time()  # we use this to sanitise our timestamps

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
                logger.warning('Failing to sustain requested framerate {0}fps for frame {1} (requested negative delay {2}s)'.format(
                    self.settings["brightfield_framerate"],
                    self.next_frame_index,
                    wait_s))
    
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
    """Emulated data capture for a set of sample brightfield frames."""
    logger.success("Initialising gater...")
    analyser = FileOpticalGater(file_source_path=settings["path"], settings=settings,)

    logger.success("Determining reference period...")
    while analyser.state != "sync":
        while not analyser.stop:
            analyser.analyze_pixelarray(analyser.next_frame(force_framerate=True))
        logger.info("Requesting user input...")
        analyser.user_select_ref_frame(10)
    logger.success(
        "Period determined ({0} frames long) and user has selected frame {1} as target.",
        analyser.pog_settings["reference_period"],
        analyser.pog_settings["referenceFrame"],
    )

    logger.success("Emulating...")
    while not analyser.stop:
        analyser.analyze_pixelarray(analyser.next_frame(force_framerate=True))

    logger.success("Plotting summaries...")
    analyser.plot_triggers()
    analyser.plot_prediction()
    analyser.plot_accuracy()
    analyser.plot_running()

    logger.success("Fin.")


if __name__ == "__main__":
    t = time.time()
    # Reads data from settings json file
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = "settings.json"

    with open(settings_file) as data_file:
        settings = json.load(data_file)

    # Performs an emulated data capture
    run(settings)
    print(time.time() - t)
