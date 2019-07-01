import time
import picamera
import numpy as np
import pandas as pd
import cv2
import io

#Capturing images to numpy arrays and pickling 

stream = io.BytesIO()

import pickle

#Number of images to capture
imnum = 11

rgb_files = {}
jpeg_files = {}

#Initialise PiCam with 640x480 res
camera = picamera.PiCamera()
camera.resolution = (640,480)

#Camera warmup time
time.sleep(2)

#Capture 11 images of same scene to a numpy array called 'rgb'
for i in range(imnum-1):    

    rgb = np.empty((480,640,3),dtype=np.uint8) #empty numpy array for rgb
    jpeg = np.empty((480,640),dtype=np.uint8) #empty numpy array for jpg
    camera.capture(rgb,'rgb')
    camera.capture(jpegs, 'jpeg')
    
    #Write each image into a dictionary 
    rgb_files["image{0}".format(i)] = rgb 
    jpegs["image{0}".format(i)] = jpeg
    
#save rgb files as a pandas data series and save to file
imageset = pd.Series(data = rgb_files)

pickle.dump(raw_files, open('test_images', 'wb'))

    

