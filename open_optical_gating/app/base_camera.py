"""From: https://github.com/miguelgrinberg/flask-video-streaming"""

import time
import threading
import sys

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident


class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """

    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                # TODO: JT writes: what happens to the client then? Does the client hang indefinitely in wait()? Is that acceptable?
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class BaseCamera(object):

    
    def __init__(self, framerate = 80, resolution = 128, shutter_speed_us = 2500, stop = False):
        """Start the background camera thread if it isn't running yet."""
        self.thread = None
        self.frame = None
        self.last_access = 0
        self.event = CameraEvent()
        
        if self.thread is None:
            self.framerate = framerate
            self.resolution = resolution
            self.shutter_speed_us = shutter_speed_us
            self.stop = stop
        
            self.last_access = time.time()

            # start background frame thread
            self.thread = threading.Thread(target = self._thread)
            self.thread.daemon = True
            self.thread.start()

            # wait until frames are available
            while self.get_frame() is None:
                time.sleep(0)
                
    def __del__(self):
        print("Camera object deleted...")
        
    def get_frame(self):
        """Return the current camera frame."""
        self.last_access = time.time()

        # wait for a signal from the camera thread
        self.event.wait()
        self.event.clear()

        return self.frame
        
    def stop_now(self):
        print("JOINING CAMERA THREAD TO MAIN...")
        self.stop = True
        time.sleep(0.4)
        self.thread.join()
        self.thread = None
        print("CAMERA THREAD JOINED...")
        

    def frames(self):
        """"Generator that returns frames from the camera."""
        raise RuntimeError("Must be implemented by subclasses.")

    def _thread(self):
        """Camera background thread."""
        print("CAMERA THREAD INITIATED...")
        frames_iterator = self.frames()
        for frame in frames_iterator:
            self.frame = frame
            self.event.set()  
            time.sleep(0.01)
            if (
                (time.time() - self.last_access > 10) 
                or self.stop
                ):
                time.sleep(0.1)
                break
        print("CAMERA THREAD STOPPED...")
        return
        #self.thread = None
        
