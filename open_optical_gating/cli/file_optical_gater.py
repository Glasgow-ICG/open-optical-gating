"""Extension of optical_gater_server for emulating gating with saved brightfield data"""

# Python imports
import sys, os, time, argparse, glob, warnings
import numpy as np
import json
import urllib.request

# Module imports
from loguru import logger
from tqdm.auto import tqdm
# See comment in pyproject.toml for why we have to try both of these:
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
        force_framerate=True
    ):
        """Function inputs:
            source                            str     A path (which may include wildcards) to a tiff file(s)
                                                       to be processed as image data.
                                                       If NULL, we will look in the settings dictionary
            settings                          dict    Parameters affecting operation
                                                      (see optical_gating_data/json_format_description.md)
            ref_frames                        arraylike
                                                      If not Null, this is a sequence of reference frames that
                                                       the caller is telling us to use from the start (rather than
                                                       optical_gater_server determining a reference sequence from the
                                                       supplied input data
            ref_frame_period                  float   Noninteger period for supplied ref frames
            repeats                           int     Number of times to play through the frames in the source .tif file
            force_framerate                   bool    Whether or not to slow down the rate at which new frames
                                                       are delivered, such that we emulate real-world speeds

        """

        # Initialise parent
        super(FileOpticalGater, self).__init__(
            settings=settings, ref_frames=ref_frames, ref_frame_period=ref_frame_period)

        self.force_framerate = force_framerate
        self.progress_bar = True  # May be updated during run_server
        # How many times to repeat the sequence
        self.repeats_remaining = repeats

        # Load the data
        self.load_data(source)

    def load_data(self, filename):
        """Load data file"""
        # Load
        logger.success("Loading image data...")
        self.data = None
        for fn in tqdm(sorted(glob.glob(filename)), desc='Loading image data'):
            logger.debug("Loading image data from file {0}", fn)
            imageData = tiffio.imread(fn)
            if len(imageData.shape) == 2:
                # Cope with loading a single image - convert it to a 1xMxN array
                imageData = imageData[np.newaxis,:,:]
            if (((imageData.shape[-1] == 3) or (imageData.shape[-1] == 4))
                and (imageData.strides[-1] != 1)):
                # skimage.io.imread() seems to automatically reinterpret a 3xMxN array as a colour array,
                # and reorder it as MxNx3. We don't want that! I can't find a way to tell imread not to
                # do that (as_grayscale does *not* do what I want...). For now I just detect it empirically
                # and undo it.
                # The test of 'strides' is an empirical one - clearly imread tweaks that to
                # reinterpret the original data in a different way to what was intended, but that
                # makes it easy to spot
                warnings.warn("Looks like imread converted a {0}-timepoint array into a colour array of shape {1}. We will fix that".format(imageData.shape[-1], imageData.shape))
                imageData = np.moveaxis(imageData, -1, 0)
            if self.data is None:
                self.data = imageData
            else:
                self.data = np.append(self.data, imageData, axis=0)
    
        if self.data is None:
            # No files found matching the pattern 'filename'
            if "example_url" in self.settings["file"]:
                if (sys.platform == "win32"):
                    os.system("color")  # Make ascii color codes work
                response = input("\033[1;31mFile {0} not found on disk. Do you want to download from the internet? [Y/n]\033[0m\n".format(filename))
                if (response.startswith("Y") or response.startswith("y") or (response == "")):
                    # Download from the URL provided in the settings file
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with tqdm(unit='B', unit_scale=True, desc="Downloading") as t:
                        urllib.request.urlretrieve(self.settings["file"]["example_url"],
                                                   filename,
                                                   reporthook=tqdm_hook(t))
                    logger.info("Downloaded file {0}".format(filename))
                    # Try again
                    self.data = tiffio.imread(filename)
                else:
                    raise
            else:
                logger.error("File {0} not found".format(filename))
                raise FileNotFoundError("File {0} not found".format(filename))

        if "decimate" in self.settings:
            # Process just every nth frame.
            # Note that I do this after all the data has loaded. While that is temporarily wasteful of memory,
            # it's the easiest way to ensure equal decimation even when the data is split across multiple files of unknown length
            self.data = self.data[::self.settings["decimate"]]
    
        self.height, self.width = self.data[0].shape

        # Initialise frame iterator and time tracker
        self.next_frame_index = 0
        self.start_time = time.time()  # we use this to sanitise our timestamps
        self.last_frame_wallclock_time = None

    def run_server(self, show_progress_bar=True):
        if show_progress_bar:
            self.progress_bar = tqdm(total=self.data.shape[0]*self.repeats_remaining, desc="Processing frames")
        super().run_server()

    def run_and_analyze_until_stopped(self):
        while not self.stop:
            self.analyze_pixelarray(self.next_frame())

    def next_frame(self):
        """ This function gets the next frame from the data source, which can be passed to analyze().
            If force_framerate is True, we will impose a delay to ensure that frames are provided
            at the rate indicated by the settings key "brightfield_framerate".
            That ensures that timings and the timestamps in plots etc are a realistic emulation
            of what would happen on a real system.
        """
        if self.force_framerate and (self.last_frame_wallclock_time is not None):
            wait_s = (1 / self.settings["brightfield"]["brightfield_framerate"]) - (
                time.time() - self.last_frame_wallclock_time
            )
            if wait_s > 1e-9:
                # the 1e-9 is a very small time to allow for the calculation
                time.sleep(wait_s - 1e-9)
            elif self.slow_action_occurred is not None:
                logger.success(
                    "File optical gater failed to sustain requested framerate {0}fps for frame {1} (requested negative delay {2}s). " \
                               "But that is no particular surprise, because we just did a {3}".format(
                        self.settings["brightfield"]["brightfield_framerate"],
                        self.next_frame_index,
                        wait_s,
                        self.slow_action_occurred
                    )
                )
            else:
                logger.warning(
                    "File optical gater failed to sustain requested framerate {0}fps for frame {1} (requested negative delay {2}s)".format(
                        self.settings["brightfield"]["brightfield_framerate"],
                        self.next_frame_index,
                        wait_s,
                    )
                )

        if self.progress_bar is not None:
            self.progress_bar.update(1)
        
        if self.next_frame_index == self.data.shape[0] - 1:
            self.repeats_remaining -= 1
            if self.repeats_remaining <= 0:
                # If this is our last frame we set the stop flag for the user/app to know
                self.stop = 'out-of-frames'
            else:
                # Start again at the first frame in the file
                self.next_frame_index = 0

        if self.force_framerate:
            # We are being asked to follow the specified framerate exactly.
            # We will do the best we can, but there will inevitably be a slight jitter in
            # the actual timings. In the spirit of real-time testing, we use the actual
            # wallclock time as the frame timestamp.
            # (We normalise by the start time, to avoid unnecessarily large numbers)
            this_frame_timestamp = time.time() - self.start_time
        else:
            # We are not being asked to follow the specified framerate exactly,
            # we are just running at whatever speed we can manage.
            # Since the analysis code may be looking at the timestamps,
            # we need to make sure they contain sane numbers
            this_frame_timestamp = self.next_frame_index / float(self.settings["brightfield"]["brightfield_framerate"])
        
        next = pa.PixelArray(
            self.data[self.next_frame_index, :, :],
            metadata={
                "timestamp": this_frame_timestamp
            },
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
        
        Note that in a settings file, if the key "input_tiff_path" is a relative path then this will be treated
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
        basename = os.path.basename(settings_file_path)
        if (basename in ["example_data_settings.json", "pi_default_settings.json"]):
            if (sys.platform == "win32"):
                os.system("color")  # Make ascii color codes work
            url = os.path.join("https://github.com/Glasgow-ICG/open-optical-gating/raw/main/optical_gating_data", basename)
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
    if ("file" in settings 
        and "input_tiff_path" in settings["file"]):
        settings["file"]["input_tiff_path"] = os.path.join(os.path.dirname(settings_file_path), settings["file"]["input_tiff_path"])

    # Provide the parsed arguments to the caller, as a way for them to access
    # any additional flags etc that they have specified
    settings["parsed_args"] = args

    return settings

# This next function taken from tqdm example code, to report progress during urlretrieve()
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
        source=settings["file"]["input_tiff_path"],
        settings=settings,
        force_framerate=True
    )

    logger.success("Running server...")
    analyser.run_server()

    logger.success("Plotting summaries...")
    analyser.plot_triggers()
    analyser.plot_prediction()
    analyser.plot_phase_histogram()
    analyser.plot_phase_error_histogram()
    analyser.plot_phase_error_with_time()
    analyser.plot_running()


if __name__ == "__main__":
    run(sys.argv[1:], "Run optical gater on image data contained in tiff file")
