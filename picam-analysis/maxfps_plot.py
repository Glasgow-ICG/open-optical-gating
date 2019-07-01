from picamera.array import PiYUVArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# testing and plotting continuous capture method to find max framerate for various resolutions 

#initiate variables
avg = [0]*19
frame_count = 0

x_axis = list(range(1,20))

camera = PiCamera(sensor_mode = 4)


for i in range(20):
    j = i+1
    camera.resolution = (j*64,j*64)
    rawCapture = PiYUVArray(camera,size=(j*64,j*64))
    
    time.sleep(0.3)
    start = time.time()
    frame_count = 0
    for frame in camera.capture_continuous(rawCapture, format = "yuv", use_video_port = True):
        image = frame.array

        frame_count+=1
        time1 = time.time()            
                
        rawCapture.truncate(0)
    
    
        if time1-start > 30:
            print(time1-start)
            avg[i-1 ] = frame_count/(time1-start)
            break
            

    #Plot process time vs x resolution
        
x_axis = [(i*100) for i in x_axis]
plt.scatter(x_axis,avg, marker = 'x', label = 'jpg')
#plt.scatter(x_axis,u_times, marker = 'x', label = 'rgb')


plt.title('Relationship between number of pixels and framerate')
plt.xlabel('Number of Pixels')
plt.ylabel('Average Framerate')

##optimizedParameters, pcov = opt.curve_fit(func,x_axis,c_times)
##plt.plot(x_axis,func(x_axis, *optimizedParameters), label = "fit")
plt.show()