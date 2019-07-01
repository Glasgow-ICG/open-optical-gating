import time
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from time import sleep
from picamera import PiCamera

#Script to find the average time to capture a still image

#initiate variables
rgb_avg = 0 
yuv_avg = 0
jpg_avg = 0
its = 20 #iterations to take mean over

xres = 32
yres = 32

rgb_times = []
yuv_times = []
jpg_times = []
x_axis = []

    #Setup Camera
camera = PiCamera()
#camera.rotation = 180
sleep(2)
    #Loop while increasing resolution
while yres < 1280:
    
    camera.resolution = (xres,yres)
    
        #Find average process time for 'its' interations
    for x in range(its-1):
    
        start = time.clock() #Start Timer

        camera.capture('jpeg.jpg', use_video_port = True)
        mid1 = time.clock() #Lap Timer
        camera.capture('rgbim','rgb',use_video_port = True)
        mid2 = time.clock()
        camera.capture('yuvim','yuv', use_video_port = True)
        end = time.clock() #End Timer
        
        jpg_avg += mid1-start
        rgb_avg += mid2-mid1
        yuv_avg += end-mid2
        
        
    jpg_avg = jpg_avg/its
    rgb_avg = rgb_avg/its
    yuv_avg = yuv_avg/its
    
    rgb_times.append(rgb_avg)
    jpg_times.append(jpg_avg)
    yuv_times.append(yuv_avg)

    x_axis.append(xres)
    
    xres += 32
    yres += 32
    
    
    
camera.close()

    #Plot process time vs x resolution
font  = {   'family' : 'normal',
            'weight' : 'normal',
            'size'   : 22}
matplotlib.rc('font', **font)
plt.scatter(x_axis,jpg_times, s = 80, c = 'c', marker = 'x', label = 'jpg')
plt.scatter(x_axis,yuv_times, s = 80, c = 'm', marker = 'x', label = 'yuv')
plt.scatter(x_axis,rgb_times, s = 80, c = 'y', marker = 'x', label = 'rgb')

plt.legend(loc = 'upper left')

plt.title('Relationship between resolution and capture time')
plt.xlabel('Resolution (X by X)')
plt.ylabel('Process Time (s)')

plt.grid(b = True, which = 'both', axis = 'both')
plt.savefig('format_times.png')

plt.show()

#print('Compressed Time: ' + str(c_avg))
#print('Uncompressed Time: ' + str(u_avg))
