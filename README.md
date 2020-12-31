# aclePIus - Open-source prospective and adaptive optical gating for 3D fluorescence microscopy of beating hearts

## Alex Drysdale, Patrick Cameron, Jonathan Taylor and Chas Nelson

### School of Physics and Astronomy, University of Glasgow, UK

Cardiac diseases account for more deaths worldwide than any other cause.
The zebrafish is a commonly used and powerful model organism for investigating cardiac conditions with a strong connection to human disease.
This is important for furthering biomedical sciences such as developing new disease models or drugs to combat those diseases.

Prospective optical gating technologies allow 3D, time-lapse microscopy of the living, beating zebrafish heart without the use of pharmaceuticals or electrical/optical pacing [1].
Further, prospective optical gating reduces the data deluge and processing time compared to other gating-based techniques.
However, currently these systems requires specialist, custom-made timing boxes and highly sophisticated system-specific software.
In order to make these tools more widely available to research groups around the world we have ported these technologies onto the popular, single-board computer, the Raspberry Pi with Python code for prospective optical gating, microscope triggering and a simple GUI for user-friendliness.

Our fully open-source system is able to perform prospective optical gating with a quality near that achieved by custom and specialised hardware/software packages. We hope that by moving this project into the open-science sphere we can ensure that all groups with an interest are able to implement prospective optical gating in a cheap (<50 EUR) and simple way.

1. Taylor, J.M., Nelson, C.J., Bruton, F.A. et al. Adaptive prospective optical gating enables day-long 3D time-lapse imaging of the beating embryonic zebrafish heart. Nature Communications 10, 5173 (2019) doi:[10.1038/s41467-019-13112-6](https://dx.doi.org/10.1038/s41467-019-13112-6)

## Installation

The following instructions have been tested on:
- Debian and CentOS Linux, Windows with Anaconda python, OS X with Anaconda python, Raspberry Pi 3 with Raspberry Pi OS (release: 2020-08-20).
- Python 3.5-3.9

### Dependencies

If you install this software through either of the following to methods, you should not need to install any extra dependencies.
All dependencies are specified in the `pyproject.toml` file.

### For users

This package is not yet published to PyPi, but you can install it directly from our git respository using `pip`.

On any platform *except* the Raspberry Pi, run the following command:

`python -m pip install --user git+https://github.com/Glasgow-ICG/open-optical-gating.git@main#egg=open-optical-gating`

(Note: use `python3` in place of `python` if you have a Python 2.x installed as well)

On the Raspberry Pi, run the following commands:

`sudo apt install libatlas-base-dev python3-gi-cairo`
`python3 -m pip install --user picamera pybase64 git+https://github.com/abdrysdale/fastpins`
`python3 -m pip install --user git+https://github.com/Glasgow-ICG/open-optical-gating.git@main#egg=open-optical-gating`

### Installation - troubleshooting

???? `python -m pip install --upgrade pip`

- On some computers it may be necessary to run all commands using `python3` instead of `python` (e.g. if there is both a python 2 and a python 3 installed)

- If you are installing in a virtual environment, the `--user` flag will lead to an error - in that case, just omit it.

- If you have a *very* old version of pip installed, you may find that the installation process is suspiciously fast and then when you try and run the code you get an error  `No module named open_optical_gating`. 
In that case the solution is to run `pip install --upgrade pip` and then follow the installation instructions again.


## Testing an installation

### Testing the install by emulating on a file

If this software is correctly installed, it should be able to run the FileOpticalGater using the example data in this repository. Run:

 `python -m open_optical_gating.cli.file_optical_gater optical_gating_data/example_data_settings.json`
 
 The first time you run, this will prompt you to download some example video data. It will then run the optical gater code on that dataset. 
During the analysis it will ask you to pick a period frame (try '10'). It will produce four output plots showing the triggers that would be sent, the predicted trigger time as the emulation went on, the accuracy of those emulated triggers and the frame processing rates.

### Testing the Raspberry Pi Triggers

If this software is correctly installed and your hardware is correctly configured (see below), you should be able to run the PiOpticalGater using the example data in this repository, from within the repository folder run

`python -m open_optical_gating/cli/check_trigger optical_gating_data/default_settings.json`

This should trigger your timing box/laser/camera depending on your configuration.
If using this to test a camera trigger, you will need to set your camera ready to recieve external triggers (see below).

### Testing the websocket interface

To test the websocket version of this software, from within the repository folder run two separate commands simultaneously (in separate terminal windows):

`python -m open_optical_gating.cli.websocket_optical_gater optical_gating_data/example_data_settings.json`
`python -m open_optical_gating.cli.websocket_example_client optical_gating_data/example_data_settings.json`

This will perform a run similar to that with file_optical_gater, but with frames being sent from the client, synchronization analysis being performed on the server, and triggers being received back by the client (which plots a crude graph at the end).


## For developers - pip installation of source code

Instead of the standard `pip install` command given above, run the following command from the directory where you want the source tree to be generated:

`python -mpip install --src "." -e git+https://github.com/Glasgow-ICG/open-optical-gating.git@main#egg=open-optical-gating`

### Tests for developers

We are currently in the process of building a pytest framework for this repository.
Watch this space.

## asclePIus - Raspberry Pi plug-n-play smart microscope

The following instructions provide a guide of how to operate the Raspberry Pi system, asclePIus, and perform various tests.
Prior to this please ensure that asclePIus has been set up correctly to control the microscope.

AsclePIus is be operated in 5V BNC trigger mode where asclePIus sends a trigger signal through pin 22 when an image capture should be performed.
Additionally, seperate pins can be used to trigger a camera at the same time if you don't have the hardware to trigger your laser and camera instantaneously.
This can be either and "edge" or "expose" trigger mode, depending on your set-up; this can be controlled in the settings JSON file.

The pin numbering system being used is the physical numbering system.

### Raspberry Pi set up

The Raspberry Pi is able to send two triggers through its GPIO pins:

* a BNC trigger that can be used as an input to a commercial microscope or laser.
* a Trigger/SYNC-A/SYNC-B connection for directly triggering many fluorescent cameras.

All pins below refer to the Raspberry Pi pin numbering (and not the GPIO number).

#### BNC trigger only

* Connect the 5V BNC cable to pin 22 and pin 6 (or any other ground pin).

#### Joint laser and fluorescence camera trigger

* Connect the laser to pin 22 and pin 6 (or any other ground pin).
* Connect the fluorescence camera in the following way:
  * Trigger: pin 8
  * SYNC-A: pin 10
  * SYNC-B: pin 12

### Triggering test

The microscope test can be run through the 'trigger_check.py' file as described above.
This is, in essence, designed to ensure that the microscope system has been set up correctly and the custom fastpins module has been installed correctly.
The test simply pulses pin 22 and pin 8 and returns an error if the software can not perform the pulsing and the user can detect a hardware if no signal is being detected by the microscope.
*Note: the triggering runs for a very very long time so make sure to quit the program*

If the microscope is not being triggered, please check the connections.
If the microscope is connected properly check all the required modules have been installed.
After this, it would be best to check if the signals are being fired by the AsclePius pins and proceed accordingly.

### Operating the timing box

1. Ensure that the settings are set to the correct values (in the `settings.json` file) specifically that

   ```'live':1```

   to capture live data. Set ```'live':0``` for an emulated data capture. If the emulation is successful the phase of the heart should resemble a rough saw tooth with the triggering times at the same phase. The prediction latency, logging and many other variables can all be set through the ```settings.json``` file. The default values are included at the end of this file.

2. Ensure the zebrafish heart is the field of view of both the fluorescence camera and the brightfield camera (PiCam). The brightfield camera can be displayed for 60 seconds using the following command. To increase the time, replace 60000 with the desired time in milliseconds.

       python -m raspivid -t 60000

    You can save the video output by using the `-o` flag, e.g.
   
       python -m raspivid -t 60000 -o file.h264

3. Launching the cli program.

       python -m open_optical_gating.cli.pi_optical_gater optical_gating_data/example_data_settings.json

4. Ensure the image capture software (QIClick for example) is ready to acquire a stack of images.

5. Select a frame from within the period to be used as the target frame (the frames are stored in a folder called 'period-data/'). You can obtain a new reference period by entering -1.  

6. The program will now attempt to capture a 3D gated image of the zebrafish heart (or other period object). The results will be stored with the image capture software.

## Settings.json defaults and meanings

```{json}
{
    "brightfield_framerate": "80",
    "brightfield_resolution": "128",
    "analyse_time": "1000",
    "awb_mode": "off",
    "exposure_mode": "off",
    "image_denoise": "0",
    "laser_trigger_pin": "22",
    "fluorescence_camera_pins": {
        "trigger": 8,
        "SYNC-A": 10,
        "SYNC-B": 12
    },
    "fluorescence_trigger_mode": "edge",
    "fluorescence_exposure_us": "1000.0",
    "frame_buffer_length": "100",
    "frame_num": "0",
    "live": "1",
    "log": "0",
    "period_dir": "period-data/",
    "prediction_latency_s": "0.015",
    "update_after_n_triggers": "5",
    "shutter_speed_us": 2500
}
```
