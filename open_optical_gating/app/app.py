#!/usr/bin/env python

"""Web app for interacting with the open optical gating system.
Much of this was inspired by https://github.com/miguelgrinberg/flask-video-streaming"""

# Imports
import os
from importlib import import_module
from flask import Flask, render_template, Response

# import camera driver
from camera_pi import Camera

#
import cli

app = Flask(__name__)


@app.route("/")
def index():
    """Video streaming home page."""
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


if __name__ == "__main__":
    app.run(debug=True, host="192.168.0.13", threaded=True)


# vc = cv2.VideoCapture(0)


# @app.route("/")
# def initialise():
#     return render_template("capture.html")


# def gen():

#     while True:

#         rval, frame = vc.read()
#         cv2.imwrite("pic.jpg", frame)
#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" + open("pic.jpg", "rb").read() + b"\r\n"
#         )


# @app.route("/video_feed")
# def video_feed():
#     return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/test")
# def tests():

#     # Defines initial variables
#     fps_max, pi_res = tb.check_fps()

#     return render_template("test.html", fps_max=fps_max, pi_res=pi_res)

# @app.route("/emulate_data_capture")
# def emulate_data_capture():
#     tb.emulate_data_capture()
#     return "Nothing"


# if __name__ == "__main__":

#     # Initialises the camera
#     # 	camera = picamera.PiCamera()
#     # 	camera.resolution = (100,100)
#     # 	camera.framerate = 24
#     # 	frame = np.empty((100,100,3), dtype=np.uint8)

#     app.run()
