# Instructions for Raspberry Pi gated microscopy

The following instructions provide a guide of how to operate the Raspberry Pi timing box (AsclePius) and perform various tests. Prior to this please ensure that AsclePius has been set up correctly to control the microscope.



Currently AsclePius can be operated in 5V BNC Only mode and Glasgow SPIM mode. Whilst in 5V BNC Only mode AsclePius sends a trigger signal through pin 22 when an image capture should be performed. Whilst in Glasgow SPIM mode AsclePius controls the laser, fluorescence camera and movement stages separately.



For all programs, please ensure that you are using Python 3 as they have not been tested on Python 2.

The pin numbering system being used is the physical numbering system.

## Set up

### 5V BNC Mode

- Connect the 5V BNC cable to pin 22 and a ground pin (pin 6 for example).



### Glasgow SPIM Mode

- Connect the laser to pin 22 and a ground pin.

- Connect the fluorescence camera in the following way:

  > - Trigger: pin 8
  > - SYNC-A: pin 10
  > - SYNC-B: pin 12

- Connect the stage controls via USB.

## Tests

### Microscope test

The microscope test can be run through the 'laser_and_camera_test.py' file. This is, in essence, designed to ensure that the microscope system has been set up correctly and the custom fastpins module has been installed correctly. The test simply pulses pin 22 and pin 8 and returns an error if the software can not perform the pulsing and the user can detect a hardware if no signal is being detected by the microscope.

*Note: the triggering runs for a very very long time so make sure to quit the program*

If the microscope is not being triggered, please check the connections. If the microscope is connected properly check all the required modules have been installed. After this, it would be best to check if the signals are being fired by AsclePius's pins and proceed accordingly.


### Stage test

The stages only apply when operating in Glasgow SPIM mode.

The stage testing can be run through the 'stage_test.py'. This operates by obtaining the addresses of all available stages and then enters an enviroment to control the stages. The stages can be tested by entering various commands.

*Note: The stages are assumed to be  a **SMC100CC/PP** and both the testing enviroment and the stage functions will need to be altered for different stages. The stage logic in both 'stage_test.py' and 'stage_control_functions.py' might also need to be updated if different.*

## Operating the timing box 

1. Ensure that the settings are set to the correct values (in the ```settings.json``` file) specifically that 

   ```'live':1``` 

   to capture live data. Set ```'live':0``` for an emulated data capture. If the emulation is successful the phase of the heart should resemble a rough saw tooth with the triggering times at the same phase. The prediction latency, logging and many other variables can all be set through the ```settings.json``` file. The default values are included at the end of this file.

2. Ensure the zebrafish heart is the field of view of both the fluorescence camera and the brightfield camera (Pi Cam). The brightfield camera can be displayed for 60s using the following command. To increase the time, replace 60 with the desired time in seconds.

   ```{bash}
   python3 camera.py 60
   ```

   You can save the video output by typing the following command.

   ```{bash}
   ./vidcap.sh NAME 60 80
   ```

   Which will save a file with name RPi-vidcap-NAME-DATE.h264 with a length of 60s at a frame rate of 80 fps.

3. Launching the timebox program.

   ```{bash}
   python3 timebox.py
   ```

   

4. Manually set the stage limits of the over which to acquire an image (only in Glasgow SPIM mode) and ensure that the stage is at the on edge of the heart. The stage will move the length of the heart from one end to another.

5. Ensure the image capture software (QIClick for example) is ready to acquire a stack of images.

6. Select a frame from within the period to be used as the target frame (the frames are stored in a folder called 'period_data//' in the same directory as the 'timebox.py' program. You can obtain a new reference period by entering -1.  

7. The program will now attempt to capture a 3D gated image of the zebrafish heart (or other period object). The results will be stored with the image capture software.



## Default settings.json values

```{json}
{"brightfield_framerate": 80,
 "brightfield_resolution":128,
 "analyse_time":1000,
 "awb_mode":"off",
 "exposure_mode":"off",
 "shutter_speed":2500,
 "image_denoise":0,
 "laser_trigger_pin":22,
 "fluorescence_camera_pins":[8,10,12],
 "usb_name":"/dev/tty/USB0",
 "usb_timeout":0.1,
 "usb_baudrate":57600,
 "usb_dataBits":8,
 "usb_parity":"N",
 "usb_XOnOff":1,
 "plane_address":1,
 "encoding":"utf-8",
 "terminators":[13,10],
 "increment":0.0005,
 "negative_limit":0,
 "positive_limit":0.075,
 "current_position":0,
 "frame_buffer_length":100,
 "frame_num":0,
 "live":0,
 "output_mode":"5V_BNC_Only",
 "log":0,
 "predictionLatency":15
 }
```

 
