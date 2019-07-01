import time
from time import sleep
from picamera import PiCamera

#basic image capturing script for compressed and uncompressed images

#Setup Camera
camera = PiCamera()
camera.rotation = 180
camera.resolution = (1920,1080)
#camera.start_preview()

sleep(1) #Camera warmup

start = time.clock()
() #Start Timer

camera.capture('compressed.jpg')
mid = time.clock() #Lap Timer
camera.capture('uncompressed.yuv', 'yuv')
end = time.clock() #End Timer

#camera.stop_preview()
camera.close()

print('Compressed Time: ' + str(mid - start))
print('Uncompressed Time: ' + str(end - mid))
