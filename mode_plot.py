import time
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from time import sleep
import picamera

#script to find average time to capture images using camera's preferred capture modes

#initiate variables
u_avg = 0
c_avg = 0

its = 4

c_times = []
u_times = []
x_axis = list(range(1,8))

for i in range(7):
    print(i)
    
    with picamera.PiCamera(sensor_mode = i) as camera:
    #camera = picamera.camera.PiCamera(sensor_mode = i)
    
    
        #Find average process time for 'its' interations
        for x in range(its):
    
            start = time.clock() #Start Timer

            camera.capture('compressed.jpg')
            mid = time.clock() #Lap Timer
            camera.capture('uncompressed','yuv')
            end = time.clock() #End Timer
        
            c_avg += mid-start
            u_avg += end-mid
            
        
        
    c_avg = c_avg/its
    u_avg = u_avg/its
    
    c_times.append(c_avg)
    u_times.append(u_avg)
    


    #Plot process time vs x resolution
plt.scatter(x_axis,c_times, marker = 'x', label = 'jpg')
plt.scatter(x_axis,u_times, marker = 'x', label = 'yuv')
plt.legend(loc = 'upper left')

plt.title('Relationship between capture mode and process time')
plt.xlabel('Capture Mode')
plt.ylabel('Capture Time (s)')

##optimizedParameters, pcov = opt.curve_fit(func,x_axis,c_times)
##plt.plot(x_axis,func(x_axis, *optimizedParameters), label = "fit")
plt.show()