# /usr/bin/python3


# Python imports
from flask import Flask, render_template, Response, request
import picamera
import numpy as np
import cv2

# Local imports
import cli as tb


# Defines app instance
app = Flask(__name__)
vc = cv2.VideoCapture(0)


@app.route("/")
def initialise():
    return render_template("capture.html")


def gen():

    while True:

        rval, frame = vc.read()
        cv2.imwrite("pic.jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + open("pic.jpg", "rb").read() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/test")
def tests():

    # Defines initial variables
    fps_max, pi_res = tb.check_fps()

    return render_template("test.html", fps_max=fps_max, pi_res=pi_res)

@app.route("/emulate_data_capture")
def emulate_data_capture():
    tb.emulate_data_capture()
    return "Nothing"


if __name__ == "__main__":

    # Initialises the camera
    # 	camera = picamera.PiCamera()
    # 	camera.resolution = (100,100)
    # 	camera.framerate = 24
    # 	frame = np.empty((100,100,3), dtype=np.uint8)

    app.run()

