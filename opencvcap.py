import time
import picamera
import numpy as np
import pandas as pd
import cv2
import io
import pickle

#capturing images to opencv object

stream = io.BytesIO()


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

    camera.capture(stream,format = 'jpeg')
    
    data = np.fromstring(stream.getvalue(),dtype = np.uint8)
    rgb = cv2.imdecode(data, 1)
    rgb = rgb[:,:,::-1]
    
    #Save each image into a dictionary 
    rgb_files["image{0}".format(i)] = rgb 
    #jpegs["image{0}".format(i)] = jpeg
    
#save rgb files as a pandas data series and save to file
imageset = pd.Series(data = rgb_files)

pickle.dump(rgb_files, open('test_images', 'wb'))

    


