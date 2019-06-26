from picamera import PiCamera
from time import sleep
import sys


#inital camera test
camera = PiCamera() 

# Sets Picam resolution
camera.resolution = (1024, 768)

# Previews the camera for a time specified as a command line argument (in seconds)
camera.rotation = 180
camera.start_preview()
sleep(float(sys.argv[1]))
camera.stop_preview()
