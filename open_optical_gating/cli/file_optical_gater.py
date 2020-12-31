"""Extension of CLI Open Optical Gating System for emulating gating with saved brightfield data"""

# Python imports
import sys, os, time, argparse
import json
import urllib.request

# Module imports
from loguru import logger
from tqdm.auto import tqdm
# See comment in pyproject.toml for why we have to try both of these
try:
    import skimage.io as tiffio
except:
    import tifffile as tiffio

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
        try:
            self.data = tiffio.imread(filename)
        except FileNotFoundError:
            if "source_url" in self.settings:
                response = input("\033[1;31mFile {0} not found on disk. Do you want to download from the internet? [Y/n]\033[0m\n".format(filename))
                if (response.startswith("Y") or response.startswith("y") or (response == "")):
                    # Download from the URL provided in the settings file
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with tqdm(unit='B', unit_scale=True, desc="Downloading") as t:
                        urllib.request.urlretrieve(self.settings["source_url"],
                                                   filename,
                                                   reporthook=tqdm_hook(t))
                    logger.info("Downloaded file {0}".format(filename))
                    # Try again
                    self.data = tiffio.imread(filename)
                else:
                    raise
            else:
                logger.error("File {0} not found".format(filename))
                raise
            

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
            elif self.justRefreshedRefFrames:
                logger.success(
                    "File optical gater failed to sustain requested framerate {0}fps for frame {1} (requested negative delay {2}s). " \
                    "But that is no particular surprise, because we just did a reference frame refresh".format(
                        self.settings["brightfield_framerate"],
                        self.next_frame_index,
                        wait_s,
                    )
                )
            else:
                logger.warning(
                    "File optical gater failed to sustain requested framerate {0}fps for frame {1} (requested negative delay {2}s)".format(
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

def load_settings(raw_args, desc, add_extra_args=None):
    '''
        Load the settings.json file containing information including
        the path to the .tif file to be processed.
        
        Params:   raw_args        list      Caller should normally pass sys.argv here
                  desc            str       Description to provide as command line help description
                  add_extra_args  function  Function describing additional arguments that argparse should expect,
                                             given the specific needs of the caller
        
        Note that in a settings file, if the key "path" is a relative path then this will be treated
        as relative to the *settings file*, not the current working directory.
        That seems the only sane behaviour, since when writing the settings file we cannot know
        what the current working directory will be when it is used.
        '''
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("settings", help="path to .json file containing settings")
    if (add_extra_args is not None):
        add_extra_args(parser)
    args = parser.parse_args(raw_args)

    # Load the file as a settings file
    settings_file_path = args.settings
    logger.success("Loading settings file {0}...".format(settings_file_path))
    try:
        with open(settings_file_path) as data_file:
            settings = json.load(data_file)
    except FileNotFoundError:
        if (os.path.basename(settings_file_path) == "example_data_settings.json"):
            url = "https://github.com/Glasgow-ICG/open-optical-gating/raw/main/examples/example_data_settings.json"
            response = input("\033[1;31mFile {0} not found on disk. Do you want to download from the internet? [Y/n]\033[0m\n".format(settings_file_path))
            if (response.startswith("Y") or response.startswith("y") or (response == "")):
                # Download from github
                os.makedirs(os.path.dirname(settings_file_path), exist_ok=True)
                urllib.request.urlretrieve(url, settings_file_path)
                with open(settings_file_path) as data_file:
                    settings = json.load(data_file)
            else:
                raise
        else:
            logger.error("File {0} not found".format(settings_file_path))
            raise
                    
    # If a relative path to the data file is specified in the settings file,
    # we will adjust it to be a path relative to the location of the settings file itself.
    # This is the only sane way to behave given that this code could be being run from any working directory
    # (Note that os.path.join correctly handles the case where the second argument is an absolute path)
    settings["path"] = os.path.join(os.path.dirname(settings_file_path), settings["path"])

    # Provide the parsed arguments to the caller, as a way for them to access
    # any additional flags etc that they have specified
    settings["parsed_args"] = args

    return settings

# This next function taken from tqdm example code
def tqdm_hook(t):
    """ Wraps tqdm instance for use with urlretrieve()    """
    last_b = [0]
    
    def update_to(b=1, bsize=1, tsize=None):
        """
            b  : int, optional
            Number of blocks transferred so far [default: 1].
            bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
            """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed
    
    return update_to


def run(args, desc):
    '''
        Run the optical gater based on a settings.json file which includes
        the path to the .tif file to be processed.
        
        Params:   raw_args   list    Caller should normally pass sys.argv here
                  desc       str     Description to provide as command line help description
    '''
    settings = load_settings(args, desc)

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
    run(sys.argv[1:], "Run optical gater on image data contained in tiff file")
