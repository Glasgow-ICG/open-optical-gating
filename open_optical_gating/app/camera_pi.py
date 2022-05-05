"""From: https://github.com/miguelgrinberg/flask-video-streaming"""

import io
import time
import picamera
from base_camera import BaseCamera
import sys

class PiCameraMonitor(picamera.PiCamera):
    def __enter__(self):
        print("PiCamera created by context manager")
        return super().__enter__()
        
    def __exit__(self, exception_type, exception_value, exception_traceback):
        print("PiCamera destroying by context manager")
        super().__exit__(exception_type, exception_value, exception_traceback)
        print("PiCamera destroyed by context manager")

class Camera(BaseCamera):        
    def frames(self):
        with PiCameraMonitor() as self.camera:
            self.camera.color_effects = (128, 128)
            
            #Apply any settings specified through base class
            if not self.framerate == None:
                self.camera.framerate = self.framerate
            if not self.resolution == None:
                self.camera.resolution = (self.resolution, self.resolution)
            if not self.shutter_speed_us == None:
                self.camera.shutter_speed = self.shutter_speed_us
            if not self.contrast == None:
                self.camera.contrast = self.contrast
            
            #Let camera warm up
            time.sleep(1)
            stream = io.BytesIO()
            for _ in self.camera.capture_continuous(stream, "jpeg", use_video_port=True):
                stream.seek(0)
                yield stream.read()
                
                #Reset before next frame
                stream.seek(0)
                stream.truncate()

