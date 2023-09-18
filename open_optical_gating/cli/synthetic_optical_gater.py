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

# Local imports
from . import file_optical_gater as server
from . import pixelarray as pa

class DynamicDrawer():
    def __init__(self, width, height, frames, noise_type, noise_level) -> None:
        """
        Drawer class used to simulate the zebrafish heart for use with open optical gating.
        Takes a model describing our phase progression and then generates frames at a given timestamp.
        Used with the SyntheticOpticalGater class.

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
            "noise_amount" : noise_level,
            "trigger_frames" : False
        }

        self.dimensions = (width, height)
        self.offset = 0 # Used for modelling fish twitching / drift
        self.phase_offset = 0

        # Initialise our canvas
        self.canvas = np.zeros((self.settings["width"], self.settings["height"]), dtype = np.uint8)

        # Motion model
        self.reset_motion_model()
        self.add_random_acceleration(5)
        """self.add_velocity_spike(5, 1, 10000)
        self.add_velocity_spike(6, 1, -10000)
        self.add_velocity_spike(7, 1, 10000)
        self.add_velocity_spike(8, 1, -10000)
        self.add_velocity_spike(9, 1, 10000)
        self.add_velocity_spike(10, 1, -10000)
        self.add_velocity_spike(11, 1, 10000)"""

        # Drift model
        self.drift_model = {
            0.0: 0,
            1000: 0
        }
        """self.drift_model = {
            0.0: 0,
            2.2: 250,
            2.5: -250,
            2.8: 0,
            3.2: 250,
            3.5: -250,
            3.8: 0
        }"""
        self.initial_drift_velocity = 0

        self.image_noise_rng = np.random.default_rng(0)

        #self.plot_motion_model()

        #self.save_video()

    def save_video(self):
        import tifffile as tf
        frames = []
        for i in range(self.settings["frames"]):
            frames.append(self.draw_frame_at_timestamp(self.get_state_at_timestamp(float(i / 200))[0])[0])

        frames = np.array(frames)
        tf.imwrite("test.tif", frames)

    # Motion model helper methods
    def reset_motion_model(self):
        self.initial_velocity = 2 * np.pi * 3 # 3 beats per second matches the zebrafish heart
        # Initialise our motion model
        self.motion_model_rng = np.random.default_rng(0)
        self.motion_model = {
            0.0: 0,
            100.0: 0
        }

    def add_random_acceleration(self, sigma = 0):
        """
        Adds a random acceleration for each timestamp in our motion model
        """
        # Set our acceleration to a random value for all times.
        for i, k in enumerate(np.linspace(0, 50, 5000)):
            self.motion_model[k] = self.motion_model_rng.normal(0, sigma)

        # Ensure our keys are in ascending order
        self.motion_model = dict(sorted(self.motion_model.items()))

    def add_velocity_spike(self, time, duration, magnitude):
        """
        Adds an acceleration spike
        """
        self.motion_model[time] = magnitude
        #self.motion_model[time + duration / 2] = -magnitude
        #self.motion_model[time + duration] = 0

        # Ensure our keys are in ascending order
        self.motion_model = dict(sorted(self.motion_model.items()))

 
    def clear_canvas(self):
        # Reinitialise our canvas
        self.canvas = np.zeros_like(self.canvas)

    def set_motion_model(self, initial_velocity, motion_model,):
        """
        Define how are timestamps is converted to phase.

        Args:
            initial_velocity (float) : Initial phase velocity of the simulation.
            motion_model (dict) : Dictionary of timestamps and accelerations.
        """

        self.initial_velocity = initial_velocity
        self.motion_model = motion_model


    def set_drift_model(self, initial_drift_velocity, drift_model):
        self.initial_drift_velocity = initial_drift_velocity
        self.drift_model = drift_model

    def get_state_at_timestamp(self, timestamp):
        """
        Uses equations of motion to determine the system state at a given timestamp with the
        drawers motion model. The function set_motion_model must be called before using this function.

        Args:
            timestamp (_type_): _description_

        Returns:
            tuple: Tuple of the timestamp, position, velocity, and acceleration
        """

        # Get our phase progression
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
                position += velocity * delta_time + 0.5 * acceleration * delta_time**2
                velocity += delta_time * acceleration

        # Get our drift
        drift_times = [*self.drift_model.keys()]
        drift_velocities = [*self.drift_model.values()]

        drift_velocity = self.initial_drift_velocity
        drift_position = 0

        drift_end = False
        for i in range(len(drift_times) - 1):
            if drift_end == False:
                # First we check if we are within the time period of interest
                if drift_times[i] <= timestamp and drift_times[i + 1] >= timestamp:
                    delta_time = timestamp - drift_times[i]
                    drift_end = True
                else:
                    delta_time = drift_times[i + 1] - drift_times[i]
                    drift_end = False

                # Next we calculate the velocity and position.
                drift_velocity = drift_velocities[i]
                drift_position += delta_time * drift_velocity

        self.offset = drift_position

        return timestamp, position, velocity, acceleration

    def set_drawing_method(self, draw_mode):
        """
        Define the method used to draw new pixels to the canvas.
        Can use np.add, np.subtract, etc. or pass a function.
        Function should take two inputs: existing canvas and an array
        to draw.

        Args:
            draw_mode (function): Function to use for drawing
        """        
        self.draw = draw_mode

    def draw_to_canvas(self, new_canvas):
        # move our current drawing canvas to the new canvas
        return self.draw(self.canvas, new_canvas)
    
    def get_canvas(self, add_noise = False, timestamp = 0):
        """
        Get the current canvas. Adds noise and ensures correct bit-depth.

        Returns:
            np.array: Canvas
        """        
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255

        # Add noise
        # For reproducibility of our synthetic data we set the random seed
        #np.random.seed(int((0 + timestamp) * 100))
        if add_noise:
            if self.settings["noise_type"] == "poisson":
                self.canvas += self.image_noise_rng.poisson(self.canvas, self.canvas.shape)
            elif self.settings["noise_type"] == "normal":
                self.canvas += self.image_noise_rng.normal(0, self.settings["noise_amount"], self.canvas.shape)
        
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
        new_canvas = self.circular_gaussian(xx + self.offset, yy + self.offset, _mean_x, _mean_y, _sdx, _sdy, _theta, _super)
        new_canvas = _br * (new_canvas / np.max(new_canvas))
        self.canvas = self.draw_to_canvas(new_canvas)

    def draw_frame_at_timestamp(self, timestamp, add_noise = True):
        """
        Draws a frame at a given timestamp.

        Args:
            phase (float): Phase to draw the frame at
        """

        phase = self.get_state_at_timestamp(timestamp)[1] % (2 * np.pi)
        self.clear_canvas()
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 32 + 8 * np.cos(phase), 32 + 8 * np.cos(phase), 0, 1.6, 256)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 26 + 8 * np.cos(phase), 26 + 8 * np.cos(phase), 0, 1.6, 256)
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 32 + 8 * np.sin(phase), 32 + 8 * np.sin(phase), 0, 1.6, 256)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 26 + 8 * np.sin(phase), 26 + 8 * np.sin(phase), 0, 1.6, 256)

        return self.get_canvas(add_noise, timestamp), phase + self.phase_offset
    

    def plot_motion_model(self):
        """
        Plot the motion model
        """
        xs = np.linspace(0, 15, 1000)
        positions = []
        velocities = []
        accelerations = []
        for x in xs:
            positions.append(self.get_state_at_timestamp(x)[1])
            velocities.append(self.get_state_at_timestamp(x)[2])
            accelerations.append(self.get_state_at_timestamp(x)[3])

        plt.figure()
        plt.title("Heart phase progression model")
        plt.plot(xs, positions, label = "Phase ($m$)")
        plt.plot(xs, velocities, label = "Phase velocity ($ms^{-1}$)")
        plt.plot(xs, accelerations, label = "Phase acceleration ($ms^{-2}$)")
        plt.legend()
        plt.show()
    
    


class SyntheticOpticalGater(server.FileOpticalGater):
    def __init__(self, settings=None):        
        super(server.FileOpticalGater, self).__init__(settings=settings)
        self.synthetic_source = DynamicDrawer(196, 196, 1000, "normal", 24)
        self.next_frame_index = 0
        self.number_of_frames = self.synthetic_source.settings["frames"]
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
        if self.synthetic_source.settings["trigger_frames"] == True:
            import tifffile as tf
            tf.imwrite(f"triggers/synthetic_fluorescence{trigger_time_s}.tif", self.synthetic_source.draw_frame_at_timestamp(trigger_time_s, add_noise = False)[0])
        return super().trigger_fluorescence_image_capture(trigger_time_s)

    def plot_future_predictions(self, prediction_s):
        from . import prospective_optical_gating as pog

        states = pa.get_metadata_from_list(self.frame_history, "states", onlyIfKeyPresent="states")
        timestamps = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent="states")
        unwrapped_phases = pa.get_metadata_from_list(self.frame_history, "unwrapped_phase", onlyIfKeyPresent="unwrapped_phase")
        first_timestamp = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent = "unwrapped_phase")[0]
        phase_offset = unwrapped_phases[0] - self.synthetic_source.get_state_at_timestamp(first_timestamp)[1]

        residuals = []
        for i, timestamp in enumerate(timestamps):
            true_phase = self.synthetic_source.get_state_at_timestamp(timestamp + prediction_s)[1] % (2 * np.pi)

            if type(self.predictor) is pog.KalmanPredictor:
                # Make a forward prediction
                estimated_phase = (self.predictor.kf.make_forward_prediction(states[i], timestamp, timestamp + prediction_s)[0][0] - phase_offset) % (2 * np.pi)
            elif type(self.predictor) is pog.IMMPredictor:
                estimated_phase = (self.predictor.imm.make_forward_prediction(states[i], timestamp, timestamp + prediction_s)[0][0] - phase_offset) % (2 * np.pi)
            elif type(self.predictor) is pog.LinearPredictor:
                estimated_phase = ((timestamp + prediction_s) * states[i][1] + states[i][0] - phase_offset) % (2 * np.pi)

            residual = estimated_phase - true_phase - (2 * np.pi)
            while residual < - np.pi:
                residual += 2 * np.pi
            residuals.append(residual)

        plt.figure()
        plt.scatter(timestamps + prediction_s, residuals)
        plt.title(f"Prediction residual at t+{prediction_s}")
        sent_trigger_times = pa.get_metadata_from_list(
            self.frame_history,
            "predicted_trigger_time_s",
            onlyIfKeyPresent="trigger_sent"
        )
        for trigger_time in sent_trigger_times:
            plt.axvline(trigger_time, ls = ":", c = "black", label = "Triggers")
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

    def plot_state_residuals(self):
        # Karlin TODO: Make this work for linear fit

        #if self.settings["prediction"]["prediction_method"] == "kalman" or self.settings["prediction"]["prediction_method"] == "IMM":
        timestamps = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent="states")
        state_estimates = pa.get_metadata_from_list(self.frame_history, "states", onlyIfKeyPresent="states")
        unwrapped_phases = pa.get_metadata_from_list(self.frame_history, "unwrapped_phase", onlyIfKeyPresent="states")
        phases = pa.get_metadata_from_list(self.frame_history, "phase", onlyIfKeyPresent="states")
        first_timestamp = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent = "states")[0]
        state_offset = np.zeros_like(state_estimates[0])
        state_offset[0] = unwrapped_phases[0] - self.synthetic_source.get_state_at_timestamp(first_timestamp)[1]
        if self.settings["prediction"]["prediction_method"] == "linear":
            state_offset[0] = unwrapped_phases[0] - self.synthetic_source.get_state_at_timestamp(first_timestamp)[1]
            state_offset[1] = state_estimates[0][1] - self.synthetic_source.get_state_at_timestamp(first_timestamp)[2]


        state_residuals = []
        measurement_residuals = []
        for i, state in enumerate(state_estimates):
            # Get the true phase at the predicted trigger time
            true_state = self.synthetic_source.get_state_at_timestamp(timestamps[i])[1:3]

            if self.settings["prediction"]["prediction_method"] == "linear":
                state[0] = unwrapped_phases[i]
                state[1] = state_estimates[i][1]

            state_residual = true_state - state + state_offset
            measurement_residual = true_state[0] - unwrapped_phases[i] + state_offset[0]

            measurement_residuals.append(measurement_residual)
            state_residuals.append(state_residual)

        state_residuals = np.array(state_residuals)
        measurement_residuals = np.array(measurement_residuals)

        print(state_residuals.shape)

        plt.figure()
        plt.title("Estimated state vs true state")
        #plt.scatter(timestamps, measurement_residuals, label = "Measured position (m)", s = 3)
        plt.scatter(timestamps, state_residuals[:, 0], label = "State position (m)", s = 10)
        plt.scatter(timestamps, state_residuals[:, 1], label = "State velocity (m/s)", s = 10)
        plt.legend()
        plt.xlabel("Timestamp (s)")
        plt.ylabel("State residual")
        plt.show()

    def get_MSE(self, prediction_s = 0.015, ignore_first_n = 0):
        """
        This is used to plot the MSE of our predictions at prediction_s seconds into the future.
        Used for tuning the Kalman filter and a useful metric to compare to linear predictor.

        Args:
            prediction_s (float): Time into the future to predict. Defaults to 0.015
            ignore_first_n (int, optional): Number of frames to ignore at the start. Used because the KF takes time to
                converge. Defaults to 0.
        """

        from . import prospective_optical_gating as pog

        states = pa.get_metadata_from_list(self.frame_history, "states", onlyIfKeyPresent="states")
        timestamps = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent="states")
        unwrapped_phases = pa.get_metadata_from_list(self.frame_history, "unwrapped_phase", onlyIfKeyPresent="unwrapped_phase")
        first_timestamp = pa.get_metadata_from_list(self.frame_history, "timestamp", onlyIfKeyPresent = "unwrapped_phase")[0]
        phase_offset = unwrapped_phases[0] - self.synthetic_source.get_state_at_timestamp(first_timestamp)[1]

        residuals = []
        for i, timestamp in enumerate(timestamps):
            if i > ignore_first_n:
                true_phase = self.synthetic_source.get_state_at_timestamp(timestamp + prediction_s)[1] % (2 * np.pi)

                if type(self.predictor) is pog.KalmanPredictor:
                    # Make a forward prediction
                    estimated_phase = (self.predictor.kf.make_forward_prediction(states[i], timestamp, timestamp + prediction_s)[0][0] - phase_offset) % (2 * np.pi)
                elif type(self.predictor) is pog.IMMPredictor:
                    estimated_phase = (self.predictor.imm.make_forward_prediction(states[i], timestamp, timestamp + prediction_s)[0][0] - phase_offset) % (2 * np.pi)
                elif type(self.predictor) is pog.LinearPredictor:
                    estimated_phase = ((timestamp + prediction_s) * states[i][1] + states[i][0] - phase_offset) % (2 * np.pi)
                elif type(self.predictor) is pog.RobustKalmanPredictor:
                    estimated_phase = 0
                else:
                    raise NotImplementedError

                residual = estimated_phase - true_phase - (2 * np.pi)
                while residual < - np.pi:
                    residual += 2 * np.pi
                residuals.append(residual)

        residuals = np.array(residuals)
        return np.mean(residuals**2)



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
    
    # Load
    def add_extra_args(parser):
        parser.add_argument("-r", "--realtime", dest="realtime", action="store_false", help="Replay in realtime (framerate as per settings file)")
    
    # General test plots
    settings = load_settings(args, desc, add_extra_args)

    # Init
    logger.success("Initialising gater...")
    analyser = SyntheticOpticalGater(settings=settings)

    # Run
    logger.success("Running server...")
    analyser.run_server()

    print(f"MSE: {analyser.get_MSE(0.015, 500)}")

    # Plot
    logger.success("Plotting summaries...")
    analyser.plot_residuals()
    analyser.plot_IMM_probabilities()
    analyser.plot_state_residuals()
    analyser.plot_future_predictions(0.015)
    analyser.plot_NIS()
    analyser.plot_likelihood()
    analyser.plot_normalised_innovation_squared()
    analyser.plot_delta_phase_phase()
    analyser.plot_triggers()
    analyser.plot_prediction()
    analyser.plot_phase_histogram()
    analyser.plot_phase_error_histogram()
    analyser.plot_phase_error_with_time()
    analyser.plot_running()

    """# Monte Carlo KF tuning
    # Here we will run optical gating multiple times to find the optimal values for our R and Q matrices
    Qs = []
    Rs = []
    MSEs = []
    NIS_means = []
    iterations = 10
    for i in range(iterations):
        print(f"\n{i}/{iterations}\n")

        np.random.seed(0)

        settings = load_settings(args, desc, add_extra_args)

        logger.success("Initialising gater...")
        analyser = SyntheticOpticalGater(settings=settings)

        logger.success("Running server...")
        analyser.run_server()

        MSEs.append(analyser.get_MSE(0, 300))
        NIS_means.append(np.mean(analyser.predictor.kf.NISs[300::]))
        Qs.append(analyser.predictor.kf.q)
        Rs.append(analyser.predictor.kf.R)

    Qs = np.array(Qs)
    Rs = np.array(Rs)
    plt.scatter(Qs, NIS_means, s = 5)
    plt.colorbar()
    plt.xlabel("Qs")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()"""

    """# Monte Carlo IMM tuning
    # Testing what the optimal ratios between our two system models is
    multipliers = []
    MSEs = []
    iterations = 15
    for i in range(iterations):

        settings = load_settings(args, desc, add_extra_args)

        logger.success("Initialising gater...")
        analyser = SyntheticOpticalGater(settings=settings)

        logger.success("Running server...")
        analyser.run_server()

        print(f"\n{i}/{iterations}\n")
        MSEs.append(analyser.get_MSE(0.015, 200))

        multipliers.append(analyser.predictor.multiplier)

    plt.figure()
    plt.scatter(multipliers, MSEs)
    plt.show()()"""

if __name__ == "__main__":
    run(sys.argv[1:], "Run optical gater on image data contained in tiff file")
