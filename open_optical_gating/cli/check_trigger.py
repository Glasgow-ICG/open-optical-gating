"""This script is used to check that the software can trigger a microscope."""

# Python imports
import time
import cli
import json
import sys


def run(args, desc, delay=500e3, number=10):
    '''
        Run the optical gater based on a settings.json file which includes
        the path to the .tif file to be processed.
        
        Params:   raw_args   list    Caller should normally pass sys.argv here
                  desc       str     Description to provide as command line help description
                  delay      float
                  number     int
        '''
    settings = load_settings(args, desc)

    laser_trigger_pin = settings["laser_trigger_pin"]
    fluorescence_camera_pins = settings["fluorescence_camera_pins"]

    duration = settings[
        "fluorescence_exposure_us"
    ]  # us, this is the duration of a laser pulse (also controls the camera exposure if set to do so)

    if cli.init_controls(laser_trigger_pin, fluorescence_camera_pins) == 0:
        # Tests laser and camera
        for i in range(number):

            cli.trigger_fluorescence_image_capture(
                delay,
                laser_trigger_pin,
                fluorescence_camera_pins,
                edge_trigger=False,
                duration=duration,
            )
    else:
        print("Could not initialise laser and fluorescence camera")

    return True


if __name__ == "__main__":
    success = run(sys.argv[1:], "Run a trigger test")
    if success:
        print("If your microscope triggered, this was successful")
    else:
        print("Something was unsuccessful.")
