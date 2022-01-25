"""This script is used to check that the software can trigger a microscope."""

# Python imports
import time
import cli
import json
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")
# logger.add("oog_{time}.log", level="DEBUG")
logger.enable("open_optical_gating")

def run(args, desc, delay=500e3, number=10):
    '''
        Run the optical gater based on a settings.json file which includes
        the path to the .tif file to be processed.
        
        Params:   raw_args   list    Caller should normally pass sys.argv here
                  desc       str     Description to provide as command line help description
                  delay      float
                  number     int
        '''
    settings = load_settings(args, desc)["trigger"]
    laser_trigger_pin = settings["laser_trigger_pin"]
    fluorescence_camera_pins = settings["fluorescence_camera_pins"]
    
    # us, this is the duration of a laser pulse (also controls the camera exposure if set to do so)
    duration = settings["fluorescence_exposure_us"]  
    
    # Create pi optical gater server
    server = pi_optical_gater.PiOpticalGater(settings=dict_data)

    # Tests laser and camera
    for i in range(number):
        t = time.time()
        server.trigger_fluorescence_image_capture(
            delay
        )
        logger.info("Took {0} s", time.time()-t)

    return True


if __name__ == "__main__":
    success = run(sys.argv[1:], "Run a trigger test")
    if success:
        print("If your microscope triggered, this was successful")
    else:
        print("Something was unsuccessful.")
