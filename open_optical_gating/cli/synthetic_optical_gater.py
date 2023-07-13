"""
Extension of optical_gater_server for emulating gating with a synthetic video source
"""

# Python imports
import sys, os, time, argparse, glob, warnings
import numpy as np
import matplotlib.pyplot as plt
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

class DynamicDrawer():
    def __init__(self, width, height, frames, noise_type, noise_level) -> None:
        """
        Drawer class used to simulate the zebrafish heart for use with open optical gating.
        Takes a model describing our phase progression and then generates frames at a given timestamp.
        Used with the SyntheticOpticalGater class.

        # TODO: Test images by sight to check that they look like they should - make sure dimensions are correct

        Args:
            width (_type_): _description_
            height (_type_): _description_
            frames (_type_): _description_
            noise_type (_type_): _description_
            noise_level (_type_): _description_
        """

        self.settings = {
            "width" : width,
            "height" : height,
            "frames" : frames,
            "noise_type" : noise_type,
            "noise_amount" : noise_level
        }

        self.dimensions = (width, height)

        self.phase_offset = 0

        # Initialise our canvas
        self.canvas = np.zeros((self.settings["width"], self.settings["height"]), dtype = np.uint8)

        # The motion model defines our heart phase progression. The key is the timestamp and the value is the acceleration at that
        # timestamp
        initial_velocity = 5
        # FIXME: Temporary fix here by setting our last timestamp in our motion model to a large number
        motion_model = {
            0: 0,
            5: 0.01,
            10: 0,
            20: 0,
            1000000: 0
        }
        self.set_motion_model(initial_velocity, motion_model)
 
    def clear_canvas(self):
        self.canvas = np.zeros_like(self.canvas)

    def set_motion_model(self, initial_velocity, motion_model):
        """
        Define how are timestamps is converted to phase.

        Args:
            initial_velocity (float) : Initial phase velocity of the simulation.
            motion_model (dict) : Dictionary of timestamps and accelerations.
        """

        self.initial_velocity = initial_velocity
        self.motion_model = motion_model

    def get_state_at_timestamp(self, timestamp):
        """
        Uses equations of motion to determine the system state at a given timestamp with the
        drawers motion model. The function set_motion_model must be called before using this function.

        Args:
            timestamp (_type_): _description_

        Returns:
            tuple: Tuple of the timestamp, position, velocity, and acceleration
        """

        # Get our timestamps and corresponding accelerations from our dictionary
        times = [*self.motion_model.keys()]
        accelerations = [*self.motion_model.values()]

        velocity = self.initial_velocity
        position = 0

        end = False
        # FIXME: This fails when we have an undefined motion model
        # We should set our acceleration to the last acceleration (or zero) in this case and still
        # return the correct values for acceleration, position, and velocity.
        for i in range(len(times) - 1):
            if end == False:
                # First we check if we are within the time period of interest
                if times[i] <= timestamp and times[i + 1] >= timestamp:
                    delta_time = timestamp - times[i]
                    end = True
                else:
                    delta_time = times[i + 1] - times[i]
                    end = False

                # Next we calculate the velocity and position.
                acceleration = accelerations[i]
                position += velocity * delta_time + 0.5 * accelerations[i] * delta_time**2
                velocity += delta_time * accelerations[i]

        return timestamp, position, velocity, acceleration

    def set_drawing_method(self, draw_mode):
        self.draw = draw_mode

    def draw_to_canvas(self, new_canvas):
        return self.draw(self.canvas, new_canvas)
    
    def get_canvas(self):
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255

        # Add noise
        if self.settings["noise_type"] == "normal":
            self.canvas += np.random.normal(0, self.settings["noise_amount"], self.canvas.shape)
        
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255
        return self.canvas.astype(np.uint8)

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

    def draw_frame_at_timestamp(self, timestamp):
        """
        Draws a frame at a given timestamp.

        Args:
            phase (float): Phase to draw the frame at
        """

        phase = self.get_state_at_timestamp(timestamp)[1] % (2 * np.pi)
        self.clear_canvas()
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 32, 32, 0, 1, 1000)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 26, 26, 0, 1, 1000)
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 32, 32, 0, 1, 1000)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 26, 26, 0, 1, 1000)

        return self.get_canvas(), phase + self.phase_offset
    


class SyntheticOpticalGater(server.FileOpticalGater):
    def __init__(self, settings=None):        
        super(server.FileOpticalGater, self).__init__(settings=settings)
        self.synthetic_source = DynamicDrawer(256, 256, 1000, "normal", 16)
        """if self.settings["brightfield"]["type"] == "gaussian":
            self.synthetic_source = Gaussian()
        elif self.settings["brightfield"]["type"] == "peristalsis":
            self.synthetic_source = Peristalsis()
        else:
            raise KeyError(f"Synthetic optical gater was given an unknown type: ({self.settings['brightfield']['type']})")"""
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

        frame_at_phase = self.synthetic_source.draw_frame_at_timestamp(this_frame_timestamp)
        next = pa.PixelArray(
            frame_at_phase[0],
            metadata={
                "timestamp": this_frame_timestamp,
                "true_phase": frame_at_phase[1]
            },
        )

        self.next_frame_index += 1

        if self.number_of_frames <= self.next_frame_index:
            self.stop = True


        return next

    def trigger_fluorescence_image_capture(self, trigger_time_s):
        return super().trigger_fluorescence_image_capture(trigger_time_s)
    
    def plot_true_predicted_phase_residual(self):
        from . import prospective_optical_gating as pog

        if type(self.predictor) is pog.KalmanPredictor or type(self.predictor) is pog.IMMPredictor:
            kf_states = pa.get_metadata_from_list(self.frame_history, "states", onlyIfKeyPresent="wait_times")
            timestamps = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent="wait_times")
            wait_times = pa.get_metadata_from_list(self.frame_history, "wait_times", onlyIfKeyPresent="wait_times")
            uwnrapped_phases = pa.get_metadata_from_list(self.frame_history, "unwrapped_phase", onlyIfKeyPresent="unwrapped_phase")
            first_timestamp = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent = "unwrapped_phase")[0]
            phase_offset = uwnrapped_phases[0] - self.synthetic_source.get_state_at_timestamp(first_timestamp)[1]
            trigger_times = timestamps + wait_times

            plot_timestamps = []
            residuals = []

            for i, state in enumerate(kf_states):
                # Get the true phase at each trigger
                true_phase_at_trigger_time = self.synthetic_source.get_state_at_timestamp(trigger_times[i])[1] % (2 * np.pi)

                # Get the Kalman filter's estimate of the phase at each trigger
                estimated_phase_at_trigger_time = (self.predictor.kf.make_forward_prediction(state, timestamps[i], trigger_times[i])[0][0] - phase_offset) % (2 * np.pi)

                # Get the residual between the two
                residual = estimated_phase_at_trigger_time - true_phase_at_trigger_time

                residuals.append(residual)
                plot_timestamps.append(timestamps[i])

            # Plot the residual
            plt.figure()
            sent_trigger_times = pa.get_metadata_from_list(
                self.frame_history,
                "predicted_trigger_time_s",
                onlyIfKeyPresent="trigger_sent"
            )
            for trigger_time in sent_trigger_times:
                plt.axvline(trigger_time, ls = ":", c = "black", label = "Triggers")
            plt.scatter(plot_timestamps, residuals, label="Residual")
            plt.xlabel("Time (s)")
            plt.ylabel("Residual (rad)")
            plt.title("True vs predicted phase residual")
            plt.show()
        elif type(self.predictor) is pog.LinearPredictor:
            kf_states = pa.get_metadata_from_list(self.frame_history, "states", onlyIfKeyPresent="wait_times")
            timestamps = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent="wait_times")
            wait_times = pa.get_metadata_from_list(self.frame_history, "wait_times", onlyIfKeyPresent="wait_times")
            uwnrapped_phases = pa.get_metadata_from_list(self.frame_history, "unwrapped_phase", onlyIfKeyPresent="unwrapped_phase")
            first_timestamp = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent = "unwrapped_phase")[0]
            phase_offset = uwnrapped_phases[0] - self.synthetic_source.get_state_at_timestamp(first_timestamp)[1]
            trigger_times = timestamps + wait_times

            plot_timestamps = []
            residuals = []

            for i, state in enumerate(kf_states):
                # Get the true phase at each trigger
                true_phase_at_trigger_time = self.synthetic_source.get_state_at_timestamp(trigger_times[i])[1] % (2 * np.pi)

                # Get the Kalman filter's estimate of the phase at each trigger
                #estimated_phase_at_trigger_time = (self.predictor.kf.make_forward_prediction(state, timestamps[i], trigger_times[i])[0][0] - phase_offset) % (2 * np.pi)
                print(state)
                estimated_phase_at_trigger_time = (state[1] + state[0] * trigger_times[i] - phase_offset) % (2 * np.pi)

                # Get the residual between the two
                residual = estimated_phase_at_trigger_time - true_phase_at_trigger_time

                residuals.append(residual)
                plot_timestamps.append(timestamps[i])

            # Plot the residual
            plt.figure()
            sent_trigger_times = pa.get_metadata_from_list(
                self.frame_history,
                "predicted_trigger_time_s",
                onlyIfKeyPresent="trigger_sent"
            )
            for trigger_time in sent_trigger_times:
                plt.axvline(trigger_time, ls = ":", c = "black", label = "Triggers")
            plt.scatter(plot_timestamps, residuals, label="Residual")
            plt.xlabel("Time (s)")
            plt.ylabel("Residual (rad)")
            plt.title("True vs predicted phase residual")
            plt.show()

    def plot_NIS(self):
        if self.settings["prediction"]["prediction_method"] == "KF":
            timestamps = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent="NIS")
            NIS = pa.get_metadata_from_list(self.frame_history, "NIS", onlyIfKeyPresent="NIS")

            plt.figure()
            plt.title("Normalised innovation squared")
            plt.scatter(timestamps, NIS)
            plt.xlabel("Timestamps (s)")
            plt.ylabel("NIS")
            plt.show()

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
    #analyser.plot_NIS()
    analyser.plot_true_predicted_phase_residual()
    #analyser.plot_likelihood()
    #analyser.plot_delta_phase_phase()
    #analyser.plot_triggers()
    #analyser.plot_prediction()
    #analyser.plot_phase_histogram()
    #analyser.plot_phase_error_histogram()
    #analyser.plot_phase_error_with_time()
    #analyser.plot_running()

if __name__ == "__main__":
    run(sys.argv[1:], "Run optical gater on image data contained in tiff file")
