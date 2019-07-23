#/usr/bin/python3


# Python imports
from flask import Flask, render_template, Response
import picamera
import numpy as np
import cv2

# Local imports
import timebox as tb


# Defines app instance
app = Flask(__name__)
vc = cv2.VideoCapture(0)

@app.route("/")
def initialise():
	return render_template('styling.html')

def gen():

	while True:

		rval, frame = vc.read()
		cv2.imwrite('pic.jpg',frame)
		yield (b'--frame\r\n'
	               b'Content-Type: image/jpeg\r\n\r\n' + open('pic.jpg','rb').read() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

	# Initialises the camera
#	camera = picamera.PiCamera()
#	camera.resolution = (100,100)
#	camera.framerate = 24
#	frame = np.empty((100,100,3), dtype=np.uint8)

	app.run()