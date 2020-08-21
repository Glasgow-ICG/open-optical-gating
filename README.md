# A fully open-source system for prospective optical gating in 3D fluorescence light sheet cardiac imaging (IN DEVELOPMENT)

## Alex Drysdale, Jonathan Taylor and Chas Nelson

### School of Physics and Astronomy, University of Glasgow, UK

Cardiac diseases account for more deaths worldwide than any other cause.
The zebrafish is a commonly used and powerful model organism for investigating cardiac conditions with a strong connection to human disease.
This is important for furthering biomedical sciences such as developing new disease models or drugs to combat those diseases. 

Prospective optical gating technologies allow 3D, time-lapse microscopy of the living, beating zebrafish heart without the use of pharmaceuticals or electrical/optical pacing [1].
Further, prospective optical gating reduces the data deluge and processing time compared to other gating-based techniques.
However, currently these systems requires specialist, custom-made timing boxes and highly sophisticated system-specific software.
In order to make these tools more widely available to research groups around the world we have ported these technologies onto the popular, single-board computer, the Raspberry Pi with Python code for prospective optical gating, microscope triggering and a simple GUI for user-friendliness.

Our fully open-source system is able to perform prospective optical gating with a quality near that achieved by custom and specialised hardware/software packages. We hope that by moving this project into the open-science sphere we can ensure that all groups with an interest are able to implement prospective optical gating in a cheap (<50 Euros) and simple way.

1. Taylor, J.M., Nelson, C.J., Bruton, F.A. et al. Adaptive prospective optical gating enables day-long 3D time-lapse imaging of the beating embryonic zebrafish heart. Nat Commun 10, 5173 (2019) doi:[10.1038/s41467-019-13112-6](https://dx.doi.org/10.1038/s41467-019-13112-6)

## Instructions for Raspberry Pi gated microscopy

The following instructions provide a guide of how to operate the Raspberry Pi timing box (AsclePius) and perform various tests. Prior to this please ensure that AsclePius has been set up correctly to control the microscope.

Currently AsclePius can be operated in 5V BNC Only mode and Glasgow SPIM mode. Whilst in 5V BNC Only mode AsclePius sends a trigger signal through pin 22 when an image capture should be performed
Whilst in Glasgow SPIM mode AsclePius controls the laser and fluorescence camera separately.

For all programs, please ensure that you are using Python 3 as they have not been tested on Python 2.

The pin numbering system being used is the physical numbering system.

## Set up

### 5V BNC Mode

- Connect the 5V BNC cable to pin 22 and a ground pin (pin 6 for example).

### Glasgow SPIM Mode

- Connect the laser to pin 22 and a ground pin.
- Connect the fluorescence camera in the following way:
  - Trigger: pin 8
  - SYNC-A: pin 10
  - SYNC-B: pin 12

## Tests

### Microscope test

The microscope test can be run through the 'trigger_check' file.
This is, in essence, designed to ensure that the microscope system has been set up correctly and the custom fastpins module has been installed correctly.
The test simply pulses pin 22 and pin 8 and returns an error if the software can not perform the pulsing and the user can detect a hardware if no signal is being detected by the microscope.

*Note: the triggering runs for a very very long time so make sure to quit the program*

If the microscope is not being triggered, please check the connections.
If the microscope is connected properly check all the required modules have been installed.
After this, it would be best to check if the signals are being fired by AsclePius's pins and proceed accordingly.

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

3. Launching the cli program.

   ```{bash}
   python3 cli.py
   ```

5. Ensure the image capture software (QIClick for example) is ready to acquire a stack of images.

6. Select a frame from within the period to be used as the target frame (the frames are stored in a folder called 'period-data//' in the same directory as the 'cli.py' program. You can obtain a new reference period by entering -1.  

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
 "fluorescence_camera_pins": {
  "trigger": 8,
  "SYNC-A": 10,
  "SYNC-B": 12
 },
 "fluorescence_trigger_mode": "edge",
 "fluorescence_exposure_us": "1000.0",
 "current_position":0,
 "frame_buffer_length":100,
 "frame_num":0,
 "live":0,
 "log":0,
 "prediction_latency":15
 }
```
