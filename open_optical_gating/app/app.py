#!/usr/bin/env python

"""Web app for interacting with the open optical gating system.
Much of this was inspired by https://github.com/miguelgrinberg/flask-video-streaming"""

# Imports
import os
from importlib import import_module
from flask import Flask, render_template, Response, request, jsonify, session
import json
from loguru import logger

# import camera driver
from camera_pi import Camera

#
import open_optical_gating.cli.cli as cli
import open_optical_gating.cli.stage_control_functions as scf

app = Flask(__name__)
app.secret_key = "test"
app.config.from_object(__name__)


# TODO make this passable arguments
settings_file = '/home/pi/open-optical-gating/examples/default_settings_with_stages.json'
# Also, can I make settings_dict global to save opening it every time


@app.route("/")
def index():
    """Video streaming home page."""

    # manual reset for session keys
    # stages = session.pop('stages')
    # pos = session.pop('positions')

    # initialise or move stages
    if 'stages' not in session.keys():
        session['stages'] = {}
        session['positions'] = {'x':0,'y':0,'z':0}
        init_hardware()
    elif request.args.get('stage') is not None:
        move_stage(request.args.get('stage'),request.args.get('size'))

    return render_template("index.html")


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame")


def init_hardware():
    # NOTE: stage currently the address, need to map those to xyz somewhere a la https://raw.githubusercontent.com/RuiSantosdotme/Random-Nerd-Tutorials/master/Projects/RPi-Web-Server/app_outputs.py
    # load current settings
    with open(settings_file) as f:
        settings_dict = json.load(f)
    
    # Initialise signallers
    # usb_serial will be one of:
    # 0 - if no failure and no usb stage
    # 1 - if fastpins fails
    # 2 - if laser pin fails
    # 3 - if camera pins fail
    # 4 - if usb stages fail
    # serial object - if no failure and usb stages desired
    usb_serial = cli.init_controls(settings_dict)
    # also inits laser and fluorescence camera triggers
    # TODO deal with errors

    # TODO add picamera settings?

    # Checks if usb_serial has recieved an error code
    if isinstance(usb_serial, int) and usb_serial > 0:
        ## TODO: How to deal with this for the app?
        logger.critical("Error code " + str(usb_serial))
        usb_serial = None
    elif isinstance(usb_serial, int) and usb_serial == 0:
        usb_serial = None
    else:
        # Defines variables for USB serial stage commands
        plane_address = stage# ?usb_information["plane_address"]
        encoding = usb_information["encoding"]
        terminator = chr(usb_information["terminators"][0]) + chr(
            usb_information["terminators"][1]
        )
        increment = usb_information["increment"]

    # store the stage details for future calls
    if usb_serial is not None:
        session['stages'] = [usb_serial, plane_address, encoding, terminator]
    # for testing only
    session['stages'] = [0, 0, 0, 0]


# The function below is executed when someone requests a URL with the stage, direction and step size in it:
def move_stage(stage, size):
    # Convert the step size into an integer
    size = int(size)


    if 'positions' in session.keys():
        # load current stage settings
        pos = session.pop('positions')

        state = scf.set_stage_to_recieve_input(session['stages'][0], stage, session['stages'][2], session['stages'][3])
        if state is 0:
            stage_result = scf.move_stage(session['stages'][0],
                                        stage,
                                        size,
                                        session['stages'][2],
                                        session['stages'][3])
        else:
            logger.critical('No response from stage {0}',stage)
        
        pos[stage] = pos[stage] + size
        session['positions'] = pos
    else:
        logger.critical('You appear to be trying to move stage {0}, which doesn\'t exist.',stage)


@app.route("/fire")
def trigger(duration=500):
    """A script to trigger the laser and camera on button press."""
    with open(settings_file) as f:
        settings_dict = json.load(f)

    laser_trigger_pin = settings_file["laser_trigger_pin"]
    fluorescence_camera_pins = settings_file["fluorescence_camera_pins"]

    cli.trigger_fluorescence_image_capture(0,laser_trigger_pin,fluorescence_camera_pins,edge_trigger=False,duration=duration)

@app.route("/parameters", methods=['GET','POST'])
def parameters():
    """Setting editor page."""
    # Original settings (need for loading and for type casting)
    with open(settings_file) as f:
        settings_dict = json.load(f)

    # If POST cast types and save
    if request.method == "POST":
        new_settings = request.form.to_dict(flat=False)
        
        to_delete = []
        for (key,val) in sorted(new_settings.items()):
            # we assume there's only one val for each
            # because there should be!
            if key in settings_dict.keys():
                print('converting type of {0} from {1} to {2}'.format(key,type(val[0]),type(settings_dict[key])))
                if isinstance(settings_dict[key],dict):
                    print('found nest {0}'.format(key))
                    new_settings[key] = {}
                    for nestkey in settings_dict[key].keys():
                        # assume only one level of nesting
                        # if additional levels are needed, make this a function and recurse
                        if nestkey in new_settings.keys():
                            print('converting type of {0} from {1} to {2}'.format(nestkey,type(new_settings[nestkey][0]),type(settings_dict[key][nestkey])))
                            new_settings[key][nestkey] = type(settings_dict[key][nestkey])(new_settings[nestkey][0])
                            print('converting type of {0} from {1} to {2}'.format(nestkey,type(new_settings[nestkey][0]),type(settings_dict[key][nestkey])))
                            if isinstance(settings_dict[key][nestkey],list):
                                new_settings[key][nestkey] = list(json.loads(new_settings[nestkey][0]))
                            else:
                                new_settings[key][nestkey] = type(settings_dict[key][nestkey])(new_settings[nestkey][0])
                elif isinstance(settings_dict[key],list):
                    new_settings[key] = list(json.loads(val[0]))
                else:
                    new_settings[key] = type(settings_dict[key])(val[0])
            else:
                print('deleting unknown key: {0}'.format(key))
                to_delete.append(key)
        for key in to_delete:
            del new_settings[key]

        settings_dict = new_settings

        with open(settings_file,'w') as f:
            json.dump(settings_dict, f, indent=2, sort_keys=True)

    return render_template("parameters.html", settings=settings_dict)


@app.route("/emulate")
def emulate():
    """Emulated run page."""
    return render_template("emulate.html")


@app.route("/run")
def run():
    """Live run page."""
    return render_template("run.html")


if __name__ == "__main__":
    app.run(debug=True, host="192.168.0.13", threaded=True)
