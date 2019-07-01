import time
import picamera
import picamera.array
import numpy as np
#from PIL import Image
import cv2
import io

def type_cap(image_formats, im_no,res = (640,480)):
    ## This function takes a list of strings of image formats e.g. ['jpg','rgb'],
    ## and im_no, the number of desired, captured images, and returns a large
    ## dictionary of the images as a 3D numpy array, with the form
    ## {'format1': {'image0': [[[X,X,X],[X,X,X],...]], 'image1': [[[...]]]}
    ##  'format2': {'image0': ...}
    
    ## Has dependencies on time, picamera, picamera.array, numpy (as np), pandas (as pd), pickle
    
    ## The default camera resolution in 640x480, but can be changed with the optional res parameter.
    
    ##Currently only works for RGB and YUV images
    
    camera = picamera.PiCamera()
    time.sleep(2)


    camera.resolution = res
    images = {}
    
    for item in image_formats:
        images[item] = {}
        if item == 'rgb':
            output = picamera.array.PiRGBArray(camera) #Set output as PiRGBArray Class
        elif item == 'yuv':
            output = picamera.array.PiYUVArray(camera)
        elif item == 'jpeg':
            
            for i in range(im_no):
                stream = io.BytesIO()
                camera.capture(stream, format = 'jpeg')
                
                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                images[item]["image%d"%i] = cv2.imdecode(data,1)
                stream.truncate(0)
                data = 0
                
            continue
                
            
        elif item == 'bayer':
            output = picamera.array.PiBayerArray(camera)
        
        
        for i in range(im_no):
        
            output.truncate(0)
            camera.capture(output, item)                   
            images[item]["image{0}".format(i)] = np.copy(output.array)
            output.truncate(0)
            
    return images

###jpg
###output = picamera.array.PiArrayOutput(camera)
##for i in range(im_no):
##    
##    camera.capture(output, 'jpeg')
##    array = output.flush()
##    jpg_images["image{0}".format(i)] = np.copy(output.array)
##    output.truncate(0)
    
#bayer
#camera.resolution = (2464,3280)
##output = picamera.array.PiBayerArray(camera)
##for i in range(im_no-1):
##    
##    camera.capture(output, 'jpeg', bayer = True)    
##    jpg_images["image{0}".format(i)] = np.copy(output.array) #save non-demoasaiced bayer to numpy array
##    output.truncate(0)
##    



