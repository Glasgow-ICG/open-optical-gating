# Open Optical Gating: open-source prospective and adaptive optical gating for 3D fluorescence microscopy of beating hearts

## Alex Drysdale, Cameron Wilson, Patrick Cameron, Jonathan Taylor and Chas Nelson.

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

If you install this software using the following methods, you should not need to manually install any extra dependencies - any required packages will be automatically installed.

### For users

This package is not yet published to PyPi, but you can install it directly from our git respository using the instructions given here.

On Windows, you will first need to install MS Visual C++ if it is not already installed on your system (see http://visualstudio.microsoft.com/visual-cpp-build-tools)

On the Raspberry Pi, you will first need to run the following two commands:

`sudo apt install libatlas-base-dev libffi-dev libssl-dev python3-gi-cairo`
`python3 -m pip install --user picamera pybase64 git+https://github.com/abdrysdale/fastpins`

Then, to install on any platform, run the following command:

`python3 -m pip install --user git+https://github.com/Glasgow-ICG/open-optical-gating.git@main#egg=open-optical-gating`

If you get the error  `python3: command not found`, substitute `python` for `python3` throughout these instructions.

If you get an error about not being able to write to `site-packages`, run the following command instead:

`python3 -m pip install --prefix=/home/pi/.local git+https://github.com/Glasgow-ICG/open-optical-gating.git@main#egg=open-optical-gating`

### Installation - troubleshooting

- `ERROR: Package u'open-optical-gating' requires a different Python: 2.7.13 not in '>=3.5,<4.0'`. 
  Rerun the installation commands substituting `python3` in place of `python`.

- Installation fails while installing dependency scikit-image:
`ModuleNotFoundError: No module named 'numpy'`
`ERROR: Command errored out with exit status 1: python3 setup.py egg_info Check the logs for full command output.`
Fix by running `python3 -m pip install numpy`, and then rerun the installation command.

- `ERROR: Can not perform a '--user' install. User site-packages are not visible in this virtualenv.` when installing inside a virtual environment. 
  Omit the `--user` flag on the `pip install` command, and rerun.

- Installation appears to succeed, but error `No module named open_optical_gating` or `ImportError: No module named 'loguru'` encountered when running the code:
    - If you are installing inside a virtual environment, rerun the install command but omit the `--user` flag. [A newer version of `pip` would have warned you about the problem - see above]
    - If you have a *very* old version of pip installed, you will see this error after a suspiciously fast installation process. Run `python3 -m pip install --upgrade pip` and then repeat the original installation instructions.

- Installation fails, with the Raspberry Pi hanging completely and requiring a reboot. This can happen if the Pi runs out of memory.
You can probably fix this by quitting all other programs that are currently running, other than the terminal window.
TeamViewer will use up most of the RAM on a Pi with 1GB RAM, so you probably need to run the initial install in person, without using TeamViewer.
(Note that TeamViewer is also likely to use too much CPU for you to be able to run live optical gating while connected over TeamViewer).

## Testing an installation

### Testing the install by emulating on a file

If this software is correctly installed, it should be able to run the FileOpticalGater using the example data in this repository. Run:

 `python3 -m open_optical_gating.cli.file_optical_gater optical_gating_data/example_data_settings.json`
 
 (Again, substitute `python` for `python3` if you need to)
 
 The first time you run, this will prompt you to download some example video data. It will then run the optical gater code on that dataset. 
During the analysis it will ask you to pick a period frame (try '10'). It will produce four output plots showing the triggers that would be sent, the predicted trigger time as the emulation went on, the accuracy of those emulated triggers and the frame processing rates.

### Testing the Raspberry Pi Triggers

If this software is correctly installed and your hardware is correctly configured (see below), you should be able to run the PiOpticalGater using the example data in this repository, from within the repository folder run

`python3 -m open_optical_gating.cli.check_trigger optical_gating_data/pi_default_settings.json`

This should trigger your timing box/laser/camera depending on your configuration.
If using this to test a camera trigger, you will need to set your camera ready to recieve external triggers (see below).

## For developers - pip installation of source code

Instead of the standard `pip install` command given above, run the following command from the directory where you want the source tree to be generated:

`python3 -m pip install --src "." -e git+https://github.com/Glasgow-ICG/open-optical-gating.git@main#egg=open-optical-gating`

As with the main install command above, you may need to add `--prefix=/home/pi/.local` on the RPi.

*Upgrading* is awkward because we aren't recording proper version numbers for our modules yet.
If you have the editable source then it should be sufficient to do a `git pull`.
If you don't have the source (which is probably the case for `optical_gating_alignment`) then you probably need to specify `--force-reinstall`.
I still need to test that. In the past I have simply gone into ` /home/pi/.local/lib/python3.7/site-packages` and manually deleted the source and egg.

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

### Generating heartbeat-synchronization trigger signals

1. Ensure that the settings are set to the correct values (in the `settings.json` file) specifically that

   ```'live':1```

   to capture live data. Set ```'live':0``` for an emulated data capture. If the emulation is successful the phase of the heart should resemble a rough saw tooth with the triggering times at the same phase. The prediction latency, logging and many other variables can all be set through the ```settings.json``` file. The default values are included at the end of this file.

2. Ensure the zebrafish heart is the field of view of both the fluorescence camera and the brightfield camera (PiCam). The brightfield camera can be displayed for 60 seconds using the following command. To increase the time, replace 60000 with the desired time in milliseconds.

       raspivid -t 60000

    You can save the video output by using the `-o` flag, e.g.
   
       raspivid -t 60000 -o file.h264

3. Launching the cli program.

       python3 -m open_optical_gating.cli.pi_optical_gater optical_gating_data/pi_default_settings.json

4. Ensure the image capture software (QIClick for example) is ready to acquire a stack of images.

5. Select a frame from within the period to be used as the target frame (the frames are stored in a folder called 'period-data/'). You can obtain a new reference period by entering -1.  

6. The program will now attempt to capture a 3D gated image of the zebrafish heart (or other period object). The results will be stored with the image capture software.
