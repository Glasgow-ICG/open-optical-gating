"""
Extension of optical_gater_server for testing KF with the SPIM.
"""

# Python imports
import sys, os, time, argparse, glob, warnings, platform
import numpy as np
import json

# Module imports
from loguru import logger
from tqdm.auto import tqdm

# Local imports
from . import optical_gater_server as server
from . import pixelarray as pa

class KalmanOpticalGater(server.OpticalGater):
    """

    """

    def __init__(
        self,
        settings=None,
        ref_frames = None,
        ref_frame_period = None
    ):
        """Function inputs:
            settings                          dict    Parameters affecting operation
                                                      (see optical_gating_data/json_format_description.md)
            ref_frames                        arraylike
                                                      If not Null, this is a sequence of reference frames that
                                                       the caller is telling us to use from the start (rather than
                                                       optical_gater_server determining a reference sequence from the
                                                       supplied input data
            ref_frame_period                  float   Noninteger period for supplied ref frames
        """

        # Initialise parent
        super(KalmanOpticalGater, self).__init__(
            settings=settings, ref_frames=ref_frames, ref_frame_period=ref_frame_period)
        
        self.stop = False
        

    def run_and_analyze_until_stopped(self):
        return None

def initialise():
    settings = load_settings("./optical_gating_data/example_data_settings.json")

    analyser = KalmanOpticalGater(
        settings = settings
    )

    logger.success("Running server...")
    analyser.run_server()

def load_settings(settings_file_path):
    '''
        Load the settings.json file
    '''

    # Load the file as a settings file
    logger.success("Loading settings file {0}...".format(settings_file_path))
    try:
        with open(settings_file_path) as data_file:
            settings = json.load(data_file)
    except FileNotFoundError:
        logger.exception("Could not find the specified settings file.")

    return settings
