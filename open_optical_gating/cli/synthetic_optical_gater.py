"""
Extension of optical_gater_server for emulating gating with a synthetic video source
"""

# Python imports
import sys, os, time, argparse, glob, warnings
import numpy as np
import json
import urllib.request
import math

# Module imports
from loguru import logger
from tqdm.auto import tqdm
# See comment in pyproject.toml for why we have to try both of these:
try:
    import skimage.io as tiffio
except:
    import tifffile as tiffio

# Local imports
#from . import optical_gater_server as server
from . import file_optical_gater as server
from . import pixelarray as pa


class SyntheticOpticalGater(server.FileOpticalGater):
    def __init__(self, settings=None):        
        super(server.FileOpticalGater, self).__init__(settings=settings)
        if self.settings["brightfield"]["type"] == "gaussian":
            self.synthetic_source = Gaussian()
        elif self.settings["brightfield"]["type"] == "peristalsis":
            self.synthetic_source = Peristalsis()
        else:
            raise KeyError(f"Synthetic optical gater was given an unknown type: ({self.settings['brightfield']['type']})")
        self.next_frame_index = 0
        self.number_of_frames = self.settings["brightfield"]["frames"]
        self.progress_bar = True  # May be updated during run_server


    def run_and_analyze_until_stopped(self):
        while not self.stop:
            self.analyze_pixelarray(self.next_frame())

    def run_server(self, show_progress_bar = True):
        if show_progress_bar:
            self.progress_bar = tqdm(total = self.number_of_frames, desc="Processing frames")
        super(server.FileOpticalGater, self).run_server()

    def next_frame(self):
        if self.progress_bar is not None:
            self.progress_bar.update(1)

        this_frame_timestamp = self.next_frame_index / float(self.settings["brightfield"]["brightfield_framerate"])

        next = pa.PixelArray(
            self.synthetic_source.draw_frame_at_phase(this_frame_timestamp * (2 * np.pi) * 3),
            metadata={
                "timestamp": this_frame_timestamp
            },
        )

        self.next_frame_index += 1

        if self.number_of_frames <= self.next_frame_index:
            self.stop = True

        return next

    def trigger_fluorescence_image_capture(self, trigger_time_s):
        return super().trigger_fluorescence_image_capture(trigger_time_s)
    

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

        # TODO: Clean this up and remove any unnecessary settings - rewrite the settings file specifically for synthetic data.
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
        if (basename in ["example_data_settings.json", "pi_default_settings.json", "synthetic_data_settings.json"]):
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
        settings["file"]["input_tiff_path"] = os.path.join(os.path.dirname(settings_file_path), os.path.expanduser(settings["file"]["input_tiff_path"]))

    # Provide the parsed arguments to the caller, as a way for them to access
    # any additional flags etc that they have specified
    settings["parsed_args"] = args

    return settings

class Drawer():
    def __init__(self, beats = 10, reference_period = 38.156, dimensions =  (256, 256)):
        """
        Base class for generating synthetic data for use with open optical gating.

        Args:
            beats (int, optional): _description_. Defaults to 10.
            reference_period (float, optional): _description_. Defaults to 38.156.
            dimensions (tuple, optional): _description_. Defaults to (256, 256).
        """        
        self.beats = beats
        self.reference_period = reference_period
        self.dimensions = dimensions

        # Settings
        # TODO: Get these from synthetic_data_settings.json
        # TODO: Add support for saving
        # TODO: Add background
        # TODO: Add different Drawer modes - set this up as base
        self.settings = {
            "dimensions" : dimensions,
            "beats" : beats,
            "reference_period" : reference_period,
            "phase_progression" : "linear", # "acceleration"
            "phase_progression_noise" : False,
            "image_noise" : "normal", # "none", "normal"
            "image_noise_amount" : 10
        }

        # Initialise our arrays
        self.sequence = np.zeros((int(math.ceil(self.settings["beats"] * self.settings["reference_period"])), *self.settings["dimensions"]), dtype = np.uint8)
        self.reference_sequence = np.zeros((int(math.ceil(self.settings["reference_period"]) + 4), *self.settings["dimensions"]), dtype = np.uint8)
        self.canvas = np.zeros(self.settings["dimensions"], dtype = np.uint8)

        # Set draw mode
        self.draw = np.add

    def generate_phase_array(self):
        return 0
 
    def clear_canvas(self):
        self.canvas = np.zeros_like(self.canvas)
        #self.canvas = self.background

    def get_canvas(self):
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255
        if self.settings["image_noise"] == "normal":
            self.canvas += np.random.normal(0, self.settings["image_noise_amount"], self.canvas.shape)
            self.canvas[self.canvas < 0] = 0
            self.canvas[self.canvas > 255] = 255
        return self.canvas.astype(np.uint8)
    
    def draw_to_canvas(self, new_canvas):
        return self.draw(self.canvas, new_canvas)
    
    def set_drawing_method(self, draw_mode):
        self.draw = draw_mode

    # Gaussian
    def circular_gaussian(self, _x, _y, _mean_x, _mean_y, _sdx, _sdy, _theta, _super):
        # Takes an array of x and y coordinates and returns an image array containing a 2d rotated Gaussian
        _xd = (_x - _mean_x)
        _yd = (_y - _mean_y)
        _xdr = _xd * np.cos(_theta) - _yd * np.sin(_theta)
        _ydr = _xd * np.sin(_theta) + _yd * np.cos(_theta)
        return np.exp(-((_xdr**2 / (2 * _sdx**2)) + (_ydr**2 / (2 * _sdy**2)))**_super)

    def draw_circular_gaussian(self, _mean_x, _mean_y, _sdx, _sdy, _theta, _super, _br):
        """
        Draw a circular Gaussian at coordinates

        Args:
            _mean_x (float): X position
            _mean_y (float): Y-position
            _sdx (float): X standard deviation
            _sdy (float): y standard deviation
            _theta (float): Angle (0-2pi)
            _super (float): Supergaussian exponent (>=0)
            _br (float): Brightness
        """
        # Draw a 2d gaussian
        xx, yy = np.indices(self.dimensions)#np.meshgrid(range(self.canvas.shape[0]), range(self.canvas.shape[1]))
        new_canvas = self.circular_gaussian(xx, yy, _mean_x, _mean_y, _sdx, _sdy, _theta, _super)
        new_canvas = _br * (new_canvas / np.max(new_canvas))
        self.canvas = self.draw_to_canvas(new_canvas)

    def draw_frame_at_phase(self, phase):
        raise NotImplementedError("Subclasses must override this function")

    def generate_reference_sequence(self):
        """
        Generate a reference sequence using the current settings.
        """        
        phase_per_frame = 2 * np.pi / self.reference_period
        phase_min = 0
        phase_max = self.reference_sequence.shape[0] * phase_per_frame

        phases = np.arange(phase_min, phase_max, phase_per_frame)
        phases = phases - phases[2]

        for i, phase in enumerate(phases):
            self.draw_frame_at_phase(phase)
            self.reference_sequence[i] = self.get_canvas()

    def generate_sequence(self):
        """
        Generate a sequence using the current settings
        """        
        # Generate a sequence of frames
        """phase_per_frame = 2 * np.pi / self.reference_period
        phase_min = 0
        phase_max = self.sequence.shape[0] * phase_per_frame

        self.phases = np.arange(phase_min, phase_max, phase_per_frame)
        self.phases = self.phases"""

        phase_min = 0
        phase_max = 2 * np.pi * self.settings["beats"]
        phases = np.linspace(phase_min, phase_max, self.sequence.shape[0])

        self.phases = [0]
        self.phase_velocities = []
        for i, phase in enumerate(phases):
            self.draw_frame_at_phase(phase)
            self.sequence[i] = self.get_canvas()
            self.phase_velocities.append(phase - self.phases[-1])
            self.phases.append(phase)

        self.phases = np.asarray(self.phases)
        self.phase_velocities = np.asarray(self.phase_velocities)

class Gaussian(Drawer):
    def draw_frame_at_phase(self, phase):
        """
        Draws a frame at a given phase. Subclass this to redefine what our sequence looks like.
        This base class uses two Gaussian blobs with a phase difference of pi/2

        Args:
            phase (float): Phase to draw the frame at
        """        
        self.clear_canvas()
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 32, 32, 0, 1, 1000)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 26, 26, 0, 1, 1000)
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 32, 32, 0, 1, 1000)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 26, 26, 0, 1, 1000)

        return self.get_canvas()


class Peristalsis(Drawer):
    """
    Single tube heart simulation

    Args:
        Drawer (class): Drawer class
    """    
    def __init__(self, beats = 10, reference_period = 38.156, dimensions =  (256, 256)):
        """
        Args:
            beats (int): Number of beats to simulate
            reference_period (float): Length of reference sequence
            dimensions ((int, int)): tuple of dimensions of simulation canvas
        """        
        super().__init__(beats, reference_period, dimensions)

        # Defines an array for our heart wall position
        self.xs = np.linspace(0, self.sequence.shape[1], 10)

    def draw_frame_at_phase(self, phase):
        self.clear_canvas()
        for i in range(self.xs.shape[0]):
            velocity_scale = 22

            self.set_drawing_method(np.add)
            _x = self.xs[i]# + 2 * np.sin(phase)
            _y = 8 + velocity_scale *  np.sin(phase / 2 + self.xs[i] / 256)**2
            _sdx = 6
            _sdy = 6
            _theta = 0
            _super = 2
            _br = 100
            self.draw_circular_gaussian(_x, _y, _sdx, _sdy, _theta, _super, _br)
            self.set_drawing_method(np.add)
            _x = self.xs[i]# + 2 * np.sin(phase)
            _y = self.dimensions[1] - (8 + velocity_scale *  np.sin(phase / 2 + self.xs[i] / 256)**2)
            _sdx = 6
            _sdy = 6
            _theta = 0
            _super = 2
            _br = 100
            self.draw_circular_gaussian(_x, _y, _sdx, _sdy, _theta, _super, _br)

        return self.get_canvas()


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
        Run the optical gater based on a settings.json
        
        Params:   raw_args   list    Caller should normally pass sys.argv[1:] here
                  desc       str     Description to provide as command line help description
    '''
    
    def add_extra_args(parser):
        parser.add_argument("-r", "--realtime", dest="realtime", action="store_false", help="Replay in realtime (framerate as per settings file)")
    
    settings = load_settings(args, desc, add_extra_args)

    logger.success("Initialising gater...")
    analyser = SyntheticOpticalGater(settings=settings)

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
