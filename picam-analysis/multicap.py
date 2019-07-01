import time
from time import sleep
from picamera import PiCamera

#test script to capture multiple images to files

#Setup Camera
camera = PiCamera()
camera.rotation = 180
camera.resolution = (1920,1080)


sleep(1) #Camera warmup

for i in range(11):

    camera.capture('comp(%d).jpg' % i)

    camera.capture( 'uncomp(%d)' % i, 'rgb')


#camera.stop_preview()
camera.close()