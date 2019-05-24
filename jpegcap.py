import time
import picamera
import numpy as np
import pandas as pd
from PIL import Image

#capturing a jpeg file to a numpy array test

jpeg = np.empty((480,640),dtype=np.uint8)

camera = picamera.PiCamera()
camera.resolution = (640,480)

time.sleep(2)

camera.capture(jpeg, 'jpg')