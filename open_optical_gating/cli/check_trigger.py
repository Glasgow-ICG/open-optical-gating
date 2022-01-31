"""This script is used to check that the software can trigger a microscope."""

# Python imports
import time
import json
import sys
from loguru import logger

from . import pi_optical_gater as piog
from . import file_optical_gater as fog

logger.remove()
logger.add(sys.stderr, level="DEBUG")
# logger.add("oog_{time}.log", level="DEBUG")
logger.enable("open_optical_gating")

def run(args, desc, delay = 5, number = 10, wait = 2):
    '''
        Run the optical gater based on a settings.json file which includes
        the path to the .tif file to be processed.
        
        Params:   raw_args   list    Caller should normally pass sys.argv here
                  desc       str     Description to provide as command line help description
                  delay      float   Time (seconds) at which trigger should be sent
                  number     int     Number of triggers to be sent
                  wait       float   Time (seconds) to wait between triggers
        '''
    # Import default pi settings and create pi optical gater server
    settings = fog.load_settings(args, desc)
    server = piog.PiOpticalGater(settings = settings)

    # Tests laser and camera
    for i in range(number):
        t = time.time()
        server.trigger_fluorescence_image_capture(
            delay
        )
        logger.info("Seding trigger {0}, took {1} s".format(i + 1, time.time()-t))
        time.sleep(wait)

    return True

if __name__ == "__main__":
    success = run(sys.argv[1:], "Run a trigger test")
    if success:
        print("If your microscope triggered, this was successful")
    else:
        print("Something was unsuccessful.")
