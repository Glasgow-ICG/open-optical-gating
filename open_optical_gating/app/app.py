#!/usr/bin/env python

"""Web app for interacting with the open optical gating system.
Much of this was inspired by https://github.com/miguelgrinberg/flask-video-streaming"""

# Imports
import os
from importlib import import_module
from flask import Flask, render_template, Response, request, jsonify, session
import json
from loguru import logger
from skimage import io
import time

# import camera driver
from camera_pi import Camera

#
import open_optical_gating.cli.cli as cli

# TODO make this passable argument
settings_file = "/home/pi/open-optical-gating/examples/example_data_settings.json"


def load_settings():
    logger.success("Settings loaded into session.")
    with open(settings_file, "r") as f:
        session["settings"] = json.load(f)


def save_settings():
    logger.success("Settings dumped into file.")
    with open(settings_file, "w") as f:
        json.dump(session["settings"], f, indent=2, sort_keys=True)


# how do I do this properly?
analyse_camera = None

app = Flask(__name__)
app.secret_key = "test"
app.before_first_request(load_settings)
app.config.from_object(__name__)


@app.route("/")
def index():
    """Video streaming home page."""
    # result will be one of:
    # 0 - if no failure
    # 1 - if fastpins fails
    # 2 - if laser pin fails
    # 3 - if camera pins fail
    result = cli.init_controls(session["settings"])
    # TODO catch failures
    return render_template("index.html")


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    # TODO: JT writes: I could really do with a comment explaining how this is working.
    # The 'yield' stuff elsewhere could do with commenting, but I understand it.
    # I don't understand what is going on here. gen() looks like it behaves as an iterator;
    # what does Response do, is it somehow permanently streaming messages for each frame, or what...?
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/fire")
def trigger(duration=500):
    """A script to trigger the laser and camera on button press."""

    laser_trigger_pin = session["settings"]["laser_trigger_pin"]
    fluorescence_camera_pins = session["settings"]["fluorescence_camera_pins"]

    cli.trigger_fluorescence_image_capture(
        0,
        laser_trigger_pin,
        fluorescence_camera_pins,
        edge_trigger=False,
        duration=duration,
    )


@app.route("/parameters", methods=["GET", "POST"])
def parameters():
    """Setting editor page."""

    # If POST, cast types and save
    if request.method == "POST":
        new_settings = request.form.to_dict(flat=False)

        to_delete = []
        for (key, val) in sorted(new_settings.items()):
            # we assume there's only one val for each
            # because there should be!
            if key in session["settings"].keys():
                logger.success(
                    "Converting type of {0} from {1} to {2}",
                    key,
                    type(val[0]),
                    type(session["settings"][key]),
                )
                if isinstance(session["settings"][key], dict):
                    logger.success("Found nest {0}", key)
                    new_settings[key] = {}
                    for nestkey in session["settings"][key].keys():
                        # assume only one level of nesting
                        # if additional levels are needed, make this a function and recurse
                        # TODO: JT writes: should this have some assertion/exception etc to guard against that scenario?
                        if nestkey in new_settings.keys():
                            logger.success(
                                "Converting type of {0} from {1} to {2}",
                                nestkey,
                                type(new_settings[nestkey][0]),
                                type(session["settings"][key][nestkey]),
                            )
                            new_settings[key][nestkey] = type(
                                session["settings"][key][nestkey]
                            )(new_settings[nestkey][0])
                            if isinstance(session["settings"][key][nestkey], list):
                                new_settings[key][nestkey] = list(
                                    json.loads(new_settings[nestkey][0])
                                )
                            else:
                                new_settings[key][nestkey] = type(
                                    session["settings"][key][nestkey]
                                )(new_settings[nestkey][0])
                elif isinstance(session["settings"][key], list):
                    new_settings[key] = list(json.loads(val[0]))
                else:
                    if val[0] == "None" or val[0] == "null":
                        logger.info("Found a None or null, converting.")
                        new_settings[key] = None
                    else:
                        new_settings[key] = type(session["settings"][key])(val[0])
            else:
                logger.info("Deleting unknown key: {0}".format(key))
                to_delete.append(key)
        for key in to_delete:
            del new_settings[key]

        session["settings"] = new_settings
        save_settings()

    return render_template("parameters.html", settings=session["settings"])


@app.route("/emulate")
def emulate():
    """Emulated run page."""
    # Global variable for analyse_camera object
    # Note: this means the developer has to manage the temporal nature of accessing this!
    # TODO: JT writes: what on earth do the above two lines of comments mean?
    global analyse_camera
    logger.debug("analyse_camera object: {0}", analyse_camera)

    # Initialise
    if request.args.get("state", False) is False:
        logger.success("Initialising")
        analyse_camera = cli.YUVLumaAnalysis(
            update_after_n_triggers=session["settings"]["update_after_n_triggers"],
            period_dir=session["settings"]["period_dir"],
        )
        logger.debug("analyse_camera object: {0}", analyse_camera)
    elif request.args.get("state", False) == "get":
        logger.success("Getting period")
        if "period" in session.keys():
            logger.info("Clearing an existing period")
            session.pop("period")
        analyse_camera.emulate_get_period(session["settings"]["path"])
        # save period in jpg for webpage
        for (i, frame) in enumerate(analyse_camera.ref_frames):
            io.imsave(
                os.path.join(
                    "open_optical_gating",
                    "app",
                    "static",
                    "period-data",
                    "{0:03d}.jpg".format(i),
                ),
                frame,
            )
        session["period"] = sorted(
            [
                os.path.join("period-data", p)
                for p in os.listdir(
                    os.path.join("open_optical_gating", "app", "static", "period-data")
                )
            ]
        )
    elif request.args.get("state", False) == "set":
        print(type(analyse_camera.ref_frames))
        logger.success(
            "Setting target frame as {0}", int(request.args.get("target", 1)) - 1
        )
        analyse_camera.state = analyse_camera.select_period(
            int(request.args.get("target", 1)) - 1
        )
        print(type(analyse_camera.ref_frames))
    elif request.args.get("state", False) == "run":
        print(type(analyse_camera.ref_frames))
        logger.success("Running emulator")
        analyse_camera.emulate()

    return render_template("emulate.html")


@app.route("/live")
def run():
    """Live run page."""
    # Global variable for analyse_camera object
    # Note: this means the developer has to manage the temporal nature of accessing this!
    # TODO: JT writes: what on earth do the above two lines of comments mean?
    global analyse_camera
    logger.debug("analyse_camera object: {0}", analyse_camera)

    # TODO: JT writes: so the "state" can be False, "get", "set" or "run"? That seems a bit weird to me. Are these states defined by the flask(?) API, or by you?
    if request.args.get("state", False) is False:
        logger.success("Initialising")
        # result will be one of:
        # 0 - if no failure
        # 1 - if fastpins fails
        # 2 - if laser pin fails
        # 3 - if camera pins fail
        result = cli.init_controls(
            session["settings"]
        )  # we re-do this in case somethings happened since doing so at index
        # TODO catch fails
        analyse_camera = cli.YUVLumaAnalysis(
            update_after_n_triggers=session["settings"]["update_after_n_triggers"],
            period_dir=session["settings"]["period_dir"],
        )
        logger.debug("analyse_camera object: {0}", analyse_camera)
    elif request.args.get("state", False) == "get":
        logger.success("Getting period")
        if "period" in session.keys():
            logger.info("Clearing an existing period")
            session.pop("period")
        analyse_camera.emulate_get_period(session["settings"]["path"])
        # save period in jpg for webpage
        for (i, frame) in enumerate(analyse_camera.ref_frames):
            io.imsave(
                os.path.join(
                    "open_optical_gating",
                    "app",
                    "static",
                    "period-data",
                    "{0:03d}.jpg".format(i),
                ),
                frame,
            )
        session["period"] = sorted(
            [
                os.path.join("period-data", p)
                for p in os.listdir(
                    os.path.join("open_optical_gating", "app", "static", "period-data")
                )
            ]
        )
    elif request.args.get("state", False) == "set":
        print(type(analyse_camera.ref_frames))
        logger.success(
            "Setting target frame as {0}", int(request.args.get("target", 1)) - 1
        )
        analyse_camera.state = analyse_camera.select_period(
            int(request.args.get("target", 1)) - 1
        )
        print(type(analyse_camera.ref_frames))
    elif request.args.get("state", False) == "run":
        print(type(analyse_camera.ref_frames))
        logger.success("Running emulator")
        analyse_camera.emulate()

    return render_template("emulate.html")


if __name__ == "__main__":
    # TODO: JT writes: I am still uncertain whether we are talking about concurrent multithreading
    # or just sequential here. I am actually now thinking that *flask* is (or at least could potentially)
    # spawn multiple *processes* to serve these requests, i.e. genuine concurrent multiprocessing.
    # My blunt question is: has anybody writing this thought through the implications of this?
    # How does that work in terms of state, race conditions etc? I'm wondering now if the "global analyse_camera"
    # line (and associated comment) was written by you guys or was in an example you have cloned.
    # I think this needs some careful thought, understanding and documenting, because I think there's
    # a risk of things getting very messy. At the very least, we need to be very clear about
    # how the threading is behaving.
    # This is where my earlier observation to Chas becomes very important, making the point that
    # this code is not implementing true REST because this code is not *stateless*.
    # That's no longer just pedantry when we are talking about concurrent multithreading,
    # it's really important!

    app.run(debug=True, host="192.168.0.13", threaded=True)
