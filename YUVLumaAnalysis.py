import picamera
import numpy as np
import j_py_sad_correlation as jps
import matplotlib.pyplot as plt
import io
from picamera import array
import time

class YUVLumaAnalysis(array.PiYUVAnalysis):
    
    #Custom class to convert and analyse Y (luma) channel of each YUV frame.
    #Extends the picamera.array.PiYUVAnalysis class, which has a stub method called analze that is overidden here.
    
    
    def __init__(self, camera, ref_frames = None, frame_num = 1, make_refs = True, size=None):
        
        #initialises relevant variables as class attributes. This is the best way i could think of to pass 
        #any external variables needed for analysis to this class (is there a better way? Not sure in Python)
        #camera = object representing PiCam,
        #frame = optiononal input to tell the object which frame in a sequence we are on        
        
        super(YUVLumaAnalysis, self).__init__(camera)
        self.frame_num = frame_num        
        self.make_refs = make_refs
        self.width, self.height = camera.resolution
        self.timer = 0 #measure how long each frame takes to capture in case this is necesary for simulator

        self.sad = np.zeros((680,20)) # arbitrary array size for testing
    

        if make_refs:
            #assuming period is 20 frames for the sake of testing. Will implement period estimation soon
            #self.ref_frames = np.empty([20, self.height, self.width], dtype = np.uint8)
             self.ref_frames = jps.doEstablishPeriodProcessingForFrame(sequence,settings) #Need to specify sequence (video or stack format?) and settings
        else:            
            self.ref_frames = ref_frames
        
    
    def analyze(self, frame):
        
        start = time.time()
        frame = frame[:,:,0] # Select just Y values from frame

        #method to analyse each frame as they are captured by the camera. Must be fast since it is running within
        #the encoder's callback, and so must return before the next frame is produced.
        
        if self.frame_num<20 and self.make_refs:            
            self.ref_frames[self.frame_num,:,:] = frame          
                        
        else:
            self.sad[self.frame_num,:] = jps.sad_with_references(frame, self.ref_frames)           
        
        self.frame_num += 1    #keeps track of which frame we are on since analyze is called for each frame
        end = time.time()
        self.timer += (end - start)
        

# These lines are for testing and will be removed at some point
x_axis = 20*[0]
frame_nums = 20*[0]
process_times = 20*[0]
total_times = 20*[0]
framerates = 20*[0]

framerate = float(input('Enter Desired Framerate: '))

for i in range(20):
    camera = picamera.PiCamera()
    res = (i+8)*32  
    camera.resolution = (res,res)
    camera.framerate = framerate
    
    output = YUVLumaAnalysis(camera)

    start = time.time()
    camera.start_recording(output, format = 'yuv')
    camera.wait_recording(10)
    camera.stop_recording()
    end = time.time()

    x_axis[i] = res
    frame_nums[i] = output.frame_num
    process_times[i] = output.timer/output.frame_num
    total_times[i] = end-start
    framerates[i] = output.frame_num/(end-start)

    print(camera.exposure_speed/1000000.0)
    output.sad = output.sad[19:output.frame_num,:]
    print(output.timer, (output.timer/output.frame_num), output.frame_num, (output.frame_num/output.timer))

    camera.close()



plt.scatter(x_axis,framerates, marker = 'x')

plt.axhline(y = 23.34, color = 'm', label = "Threshold Frame Rate")
plt.axvline(x = 300, color = 'c', label = "Minimum Image Size")
plt.axvline(x = 525, color = 'y', label = "Current Image Size")
plt.legend(loc = 'upper right')


plt.title('Relationship between resolution and Frame Rate')
plt.xlabel('Resolution (X by X)')
plt.ylabel('Frame Rate (fps)')

plt.grid(b = True, which = 'both', axis = 'both')
plt.savefig('SADTimes.png')
plt.show()
