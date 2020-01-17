# Instructions for 3D gated light sheet microscopy


The following instructions provide a guide of how to operate the timing box (AsclePius) and perform various tests. Prior to this please ensure that AsclePius has been set up correctly to control the laser, fluorescence camera and movement stages.

## Tests

### Laser and Camera test

The laser and camera test can be run through the 'laser_and_camera_test.py' file. The program triggers the laser and camera to obtain a fluorescence image. The user can then check whether the laser and camera is being triggered successfully.

*Note: the triggering runs for a very very long time so make sure to quit the program*

If the laser or camera are not being triggered, please check the connections. If the laser and camera are connected proberly check all the required modules have been installed. After this, it would be best to check if the signals are being fired by AsclePius's pins and proceed accordingly.


### Stage test

The stage testing can be run through the 'stage_test.py'. This operates by obtaining the addresses of all available stages and then enters an enviroment to control the stages. The stages can be tested by entering various commands.

*Note: The stages are assumed to be **STAGE MODEL** and both the testing enviroment and the stage functions will need to be altered for different stages. The stage logic in both 'stage_test.py' and 'stage_control_functions.py' might also need to be updated if different.*

## Operating the timing box 

1. Ensure the fish heart (or the period object being imaged) is the field of view of both the fluorescence camera and the brightfield camera (Pi Cam).
2. Launching the timebox program.
>'python3 timebox.py'
3. Manually set the stage limits of the over which to acquire an image.
4. Ensure the image capture software (QIClick for example) is ready to acquire a stack of images.
5. Select a frame from within the period to be used as the target frame (the frames are stored in a folder called 'period_data//' in the same directory as the 'timebox.py' program. You can obtain a new reference period by entering -1.  
6. The program will now attempt to capture a 3D gated image of the zebrafish heart (or other period object). The results will be stored with the image capture software.



# To-dos

- [ ] Explain logic of stages.
- [ ] Enter stage commands.
- [ ] Create connection diagram for laser and camera.
- [ ] Explain how to view PiCam display and to save output (refer to official guide for more info).
- [ ] Emulator mode vs live mode
- [ ] Alter stage limits code so can manually be set.
- [ ] Display video whilst setting stage limits.
