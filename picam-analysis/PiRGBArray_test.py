import time
import picamera
import picamera.array
import numpy as np
import pandas as pd
import pickle

# test script to investigate how picamera.array classes and methods work

im_no = 5

#Individual dictionaries for each filetype
##rgb_images = {}
##jpg_images = {}
##yuv_images = {}
##bay_images = {}

#Master dictionary to store all arrays
#images = {}

#camera = picamera.PiCamera()
#time.sleep(2)


#camera.resolution = (640,480)

#Capture sets of images of different filetypes as arrays
start = time.clock()
#RGB

def type_capture(list, im_no):
    camera = picamera.PiCamera()
    time.sleep(2)


    camera.resolution = (640,480)
    images = {}
    
    for item in list:
        if item == 'rgb':
            output = picamera.array.PiRGBArray(camera) #Set output as PiRGBArray Class
        elif item == 'yuv':
            output = picamera.array.PiYUVArray(camera)
        elif item == 'jpeg':
            output = picamera.array.PiArrayOutput(camera)
    #elif format == 'bayer':
        #output = picamera.array.PiBayerArray(camera)
        images[item] = {}
        
        for i in range(im_no):
    
            camera.capture(output, item)       
            #rgb_images["image{0}".format(i)] = np.copy(output.array) #Copy output into 3D numpy array
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

#yuv
##output = picamera.array.PiYUVArray(camera)
##for i in range(im_no):
##    
##    camera.capture(output, 'yuv')    
##    yuv_images["image{0}".format(i)] = np.copy(output.array)
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

images = type_capture(['rgb', 'yuv'], 5)

end = time.clock()
print(end-start)

##images["rgb"] = rgb_images
###images["jpg"] = jpg_images
##images["yuv"] = yuv_images
###images["bay"] = bay_images

images = pd.Series(data = images)
pickle.dump(images, open('image_set', 'wb'))


