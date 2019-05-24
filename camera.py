from picamera import PiCamera
from time import sleep

#inital camera test

camera = PiCamera()
camera.rotation = 180

camera.start_preview()
sleep(10)
camera.stop_preview()
