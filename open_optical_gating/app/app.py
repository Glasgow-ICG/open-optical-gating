#!/usr/bin/env python

"""Web app for interacting with the open optical gating system.
Much of this was inspired by https://github.com/miguelgrinberg/flask-video-streaming"""

# Imports
import os
from importlib import import_module
from flask import Flask, render_template, Response, request, jsonify
import json

# import camera driver
from camera_pi import Camera

#
import open_optical_gating.cli as cli

app = Flask(__name__)


# TODO make this passable arguments
settings_file = '/home/pi/open-optical-gating/examples/default_settings_with_stages.json'


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


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
        for (key,val) in new_settings.items():
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
            json.dump(settings_dict, f, indent=2)

    return render_template("parameters.html", settings=settings_dict)


@app.route("/check")
def check():
    """Page to check triggering and stages."""
    return render_template("check.html")


@app.route("/emulate")
def emulate():
    """Emulated run page."""
    return render_template("emulate.html")


@app.route("/run")
def run():
    """Live run page."""
    return render_template("run.html")


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/parameters", methods=['POST'])
def save_settings():
    parameters = request.form['parameters']

    with open(settings_file+'_new','w') as f:
        json.dump(parameters, f)
    return None


if __name__ == "__main__":
    app.run(debug=True, host="192.168.0.13", threaded=True)
