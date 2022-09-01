#!/usr/bin/env python

"""Web app for interacting with the open optical gating system.
Much of this was inspired by https://github.com/miguelgrinberg/flask-video-streaming"""

# Imports
import sys, io, os, glob, shutil, cgi, json, time, cv2, imageio, subprocess
import urllib.request
import numpy as np
from importlib import import_module
from multiprocessing import Process, Queue
from zipfile import ZipFile
from flask import Flask, render_template, Response, request, jsonify, session, stream_with_context, redirect, url_for
from tifffile import imread

# Local Imports	
from camera_pi import Camera
import open_optical_gating.cli.pi_optical_gater_app as sync
import retrospective_log_scraping.scrape_and_plot as scrape

# Initialise Flask App
app = Flask(__name__)
app.secret_key = "test"
app.config.from_object(__name__)
app.config["CACHE_TYPE"] = 'null'

# Import our optical gating settings and initialise Pi Object 
settings_file = "/home/pi/open-optical-gating/optical_gating_data/pi_settings.json"
with open(settings_file) as json_file:
    settings = json.load(json_file)
syncObject = sync.PiOpticalGater(settings = settings)

@app.after_request
def add_header(response):
    """
    Prevent the caching of post-sync figures by browser.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Cache-Control"] = "public, max-age = 0"
    return response
    
@app.route("/settings-page/")
def settings_page():
    """
    A page to tune the settings of the Pi Camera for desired parameters (resolution, contrast, 
    brightness, etc) as well as sending testing triggers to the microscope.
    """
    previousPageSettings = True
    # Setup the trigger pins
    syncObject.setup_pins()
    
    #Render the index page with the required arguments for the settings pane
    return render_template(
        "settings.html",    
        brightfield_framerate = settings["brightfield"]["brightfield_framerate"],
        brightfield_resolution = settings["brightfield"]["brightfield_resolution"],
        shutter_speed_us = settings["brightfield"]["shutter_speed_us"],
        contrast = settings["brightfield"]["contrast"]
        )
   
@app.route("/video_feed")
def video_feed():
    """Video streaming function."""
    
    def gen_frame(camera):
        """Video streaming generator function."""
        while not camera.stop:
            frame = camera.get_frame()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            
    # Camera defined globally so it can be stopped by other functions
    global camera
    camera = Camera(
        framerate = settings["brightfield"]["brightfield_framerate"],
        resolution = settings["brightfield"]["brightfield_resolution"],
        shutter_speed_us = settings["brightfield"]["shutter_speed_us"],
        contrast = settings["brightfield"]["contrast"]
    )
    
    if not camera.stop:
        return Response(
            gen_frame(camera), 
            mimetype="multipart/x-mixed-replace; boundary=frame"
            )
    else:
        print("Video feed broken...")
        return 

@app.route("/send-trigger/")
def send_trigger():
    """Calls pi_optical_gater trigger function to send a trigger to user microscope."""
    # Send a trigger in 0.001 seconds
    syncObject.trigger_fluorescence_image_capture(0.001)
    camera.stop = True
    time.sleep(0.2)
    return settings_page()
    
@app.route("/update-setting/", methods = ["POST"])
def update_setting():
    """
        A function which is called when the framerate html number input is engaged.
        Camera is stopped in order to be initialised with new settings.
    """
    # Get the requested change
    changeList = list(request.form.items())
    
    # Update the relevant setting in the settings dict
    settings["brightfield"][changeList[0][0]] = int(changeList[0][1])
    
    # Dump the updated settings to json
    with open(settings_file, "w") as fp:
        json.dump(settings, fp, indent = 4)
    
    # Stop the camera
    camera.stop = True
    
    # Wait for concurrent processes to complete before rendering the settings page
    time.sleep(0.2)
    return settings_page()
    
@app.route("/live-feed-live/")
def live_feed_live():
    """
    Currently poorly-functioning attempt to get a live-feed to display on the 'Sync'
    page. Designed to read a frame from the frameQueue (filled in pi_optical_gater_app).
    This is converted from YUV to JPEG format, and yielded similar to the 'Settings'
    page live feed.
    Currently, threading seems to have reached it's limit, so the frames (however long the buffer is)
    only stream once other threads have been stopped.
    """
    def gen_frame_live():
        while stopQueue.empty():
            # Read the frame from the frameQueue
            frameYUV = frameQueue.get()
            # Encode the frame to jpg and yield
            ret, buffer = cv2.imencode(".jpg", frameYUV)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        # Sleep to let concurrent processes/threads complete
        time.sleep(0.1)
    return Response(
        gen_frame_live(), 
        mimetype="multipart/x-mixed-replace; boundary=frame"
        )

def stream_template(template_name, **context):
    """Stream contents to template with context."""
    app.update_template_context(context)
    template = app.jinja_env.get_template(template_name)
    stream = template.generate(context)
    return Response(stream_with_context(stream))

@app.route("/reboot/")
def reboot():
    """
    Convoluted way to reboot the pi when the reboot button has been pressed 
    while also redirecting the user back to the index page.
    """
    # Create a global variable to be detected by the index function
    global recent
    recent = 0
    return redirect(url_for("index"))
    
@app.route("/print-data/", methods = ['POST'])
def print_data():
    """Streams live sync readouts to the 'Sync' web-server page."""
    # Read the reference frame choice from the user input
    changeList = list(request.form.items())
    refChoice = changeList[0][1]
    refSelectQueue.put(changeList[0][1])
    # Define live data generator
    def g():
        """Yield data."""
        while stopQueue.empty():
            # Only proceed if the queue has not already been read 
            if not eventQueue.empty():
                queueContents = eventQueue.get()
                timeFloat = queueContents[0]
                timeRounded = int(timeFloat * (10**1))/(10**1)
                timeNow = str(timeRounded)
                framerateNow = str(queueContents[1])
                triggersNow = str(queueContents[2])
                stateNow = str(queueContents[3])
                yield timeNow, triggersNow, framerateNow, stateNow
            # Sleep to release GIL
            time.sleep(0.001)
        # Sleep to allow concurrent processes to complete 
        time.sleep(0.1)
        print("DATA  PRINTER SHUT DOWN COMPLETELY...")
        
    return stream_template(
        "live.html", 
        data = g(),
        index_setting = "Running",
        time_limit_seconds = settings["general"]["time_limit_seconds"], 
        trigger_limit = settings["general"]["trigger_limit"]
    )

@app.route("/")
def index(index_setting = "Setup"):
    """Setup for the live run page."""
    # Convoluted way of rebooting the pi if the user clicks reboot
    # or the user was previously on the settings page
    if (
        "camera" in globals()
        or "recent" in globals()
        ):
        os.system("sudo reboot")
        
    # Initialise global queues to send/recieved data from the concurrent sync process
    global eventQueue
    eventQueue = Queue()
    global stopQueue
    stopQueue = Queue()
    global refActivateQueue 
    refActivateQueue = Queue()
    global refSelectQueue
    refSelectQueue = Queue()
    
    # Delete all temporary reference sequence data from the static folder
    all_image_paths = sorted(glob.glob("/home/pi/open-optical-gating/open_optical_gating/app/static/img_*.jpeg"))
    for image_path in all_image_paths:
        os.remove(image_path)
    print("TEMPORARY DATA DELETED...")
    
    return render_template(
        "live.html",
        time_limit_seconds = settings["general"]["time_limit_seconds"], 
        trigger_update_number = settings["reference"]["update_after_n_criterions"],
        trigger_limit = settings["general"]["trigger_limit"],
        shutter_speed_us = settings["brightfield"]["shutter_speed_us"],
        contrast = settings["brightfield"]["contrast"],
        framerate = settings["brightfield"]["brightfield_framerate"],
        save_first_n_frames = settings["brightfield"]["save_first_n_frames"],
        log_level = settings["general"]["log_level"],
        triggers_per_timelapse = settings["reference"]["triggers_between_timelapse"],
        timelapse_pause = settings["general"]["pause_for_timelapse"],
        min_time_between_triggers = settings["general"]["min_time_between_triggers"],
        index_setting = index_setting
        )

@app.route("/update-setting-live/", methods = ["POST"])
def update_setting_live():
    """
    Update the relevant settings as prompted by the user via the web-server's 'Sync' pane.
    """
    # Get the requested change
    changeList = list(request.form.items())
    
    # Altering the limited settings available to the user
    # This could be streamlined by changing the structure of the settings json
    if changeList[0][0] == "time_limit_seconds":
        settings["general"]["time_limit_seconds"] = int(changeList[0][1])
    elif changeList[0][0] == "trigger_limit":
        settings["general"]["trigger_limit"] = int(changeList[0][1])
    elif changeList[0][0] == "update_after_n_criterions":
        settings["reference"]["update_after_n_criterions"] = int(changeList[0][1])
    elif changeList[0][0] == "shutter_speed_us":
        settings["brightfield"]["shutter_speed_us"] = int(changeList[0][1])
    elif changeList[0][0] == "contrast":
        settings["brightfield"]["contrast"] = int(changeList[0][1])
    elif changeList[0][0] == "framerate":
        settings["brightfield"]["brightfield_framerate"] = int(changeList[0][1])
    elif changeList[0][0] == "save_first_n_frames":
        settings["brightfield"]["save_first_n_frames"] = int(changeList[0][1])
    elif changeList[0][0] == "log_level":
       settings["general"]["log_level"] = changeList[0][1]
       print(changeList[0][1])
    elif changeList[0][0] == "triggers_per_timelapse":
       settings["reference"]["triggers_between_timelapse"] = int(changeList[0][1])
    elif changeList[0][0] == "timelapse_pause":
       settings["general"]["pause_for_timelapse"] = int(changeList[0][1])
    elif changeList[0][0] == "min_time_between_triggers":
       settings["general"]["min_time_between_triggers"] = float(changeList[0][1])

    # Dump the updated settings to json
    with open(settings_file, "w") as fp:
        json.dump(settings, fp, indent = 4)
       
    return index() 

@app.route('/activate-ref/')
def activate_ref_input():
    """
    Called when pi_optical_gater_app.py in a child process places an item in refActivateQueue.
    Reads the initial reference sequence from the ref_sequence_folder, converts to jpeg,
    and moves them to static.
    Live.html is rendered with the relevant file paths and image numbers.
    """
    # Find the most recent reference sequence folder
    print("prompted for reference selection")
    ref_sequence_length = refActivateQueue.get()
    all_folders = glob.glob("/home/pi/open-optical-gating/open_optical_gating/app/reference_sequences/*")
    most_recent_folder = max(all_folders, key = os.path.getctime)
    
    # Iterate through each image in the ref sequence folder, convert to jpeg, move to static
    ref_frame_paths = []
    for i, image_path in enumerate(sorted(glob.glob(most_recent_folder + "/*"))):
        image = imread(image_path)
        imageio.imsave("/home/pi/open-optical-gating/open_optical_gating/app/static/img_{0}.jpeg".format(str(i).zfill(3)), image)
        ref_frame_paths.append([i, "img_{0}.jpeg".format(str(i).zfill(3))])

    return render_template(
        "live.html", 
        index_setting = "PickRefFrame", 
        ref_sequence_length = ref_sequence_length, 
        ref_frame_paths = ref_frame_paths
    )

@app.route("/start-live/")
def start_live():
    print("SYNC INITIATED...")
    
    # Initialise a sync object
    syncObject = sync.PiOpticalGater(settings = settings)
    
    # Setup the appropriate trigger pins
    syncObject.setup_pins()
    
    # Assign the start_sync function to a new process and start
    global syncProcess
    syncProcess = Process(target = syncObject.start_sync, args = (eventQueue, stopQueue, refActivateQueue, refSelectQueue, ))
    syncProcess.start()
    
    # A while loop to keep the main process active
    while stopQueue.empty():
        # Sleep to release GIL
        time.sleep(0.0001)
        # Once pi_optical_gater_app requests a reference frame, redirect to activate_ref_input
        if not refActivateQueue.empty():
            return redirect(url_for("activate_ref_input"))

@app.route("/stop-live/")
def stop_live():
    """Stop live sync."""
    # Place True in the global stopQueue
    stopQueue.put(True)
    
    # Sleep to wait for all processes/threads to complete before rendering the page
    time.sleep(0.2)
    print("SHUTTING DOWN PI PROCESS...")
    
    # Terminate and join the syncProcess
    syncProcess.terminate()
    syncProcess.join()
    
    print("PI PROCESS SHUT DOWN AND SYNC COMPLETE...")
    
    return index()

@app.route("/post-sync-setup/")
def post_sync_setup():
    """A page to display relevant plots generated from the logs."""
    
    with open(settings_file) as json_file:
        settings = json.load(json_file)
    
    # Find the most recent log file in the user_log_folder
    all_logs = glob.glob("/home/pi/user_log_folder/*")
    most_recent_log = max(all_logs, key = os.path.getctime)
    
    all_vids = glob.glob("/home/pi/open-optical-gating/open_optical_gating/app/reference_sequences/VID*")
    most_recent_vid = max(all_vids, key = os.path.getctime)
    print(most_recent_vid)
    
    with ZipFile( "/home/pi/open-optical-gating/open_optical_gating/app/static/first_n_frames.zip", "w") as zipObj:
        for imagePath in glob.glob(most_recent_vid + "/*.tiff"):
            zipObj.write(imagePath, os.path.basename(imagePath))
    
    # If the log is a real log (not empty) we move it to the static folder, overwriting the previous log
    if len(open(most_recent_log).readlines()) > 5:
        os.rename(most_recent_log, "/home/pi/open-optical-gating/open_optical_gating/app/static/most_recent_log.log")
        
    # Run the log scraper on this log file
    scrape.run("/home/pi/open-optical-gating/open_optical_gating/app/static/most_recent_log.log", '/home/pi/open-optical-gating/open_optical_gating/app/retrospective_log_scraping/log_keys.json')
    
    # Zip all generated plots in the static folder to be downloaded by the user
    with ZipFile( "/home/pi/open-optical-gating/open_optical_gating/app/static/plots.zip", "w") as zipObj:
        for plotPath in glob.glob("/home/pi/open-optical-gating/open_optical_gating/app/static/*.jpg"):
            zipObj.write(plotPath, os.path.basename(plotPath))
            
    refSequenceFolders = []
    for refFolderName in glob.glob("/home/pi/open-optical-gating/open_optical_gating/app/reference_sequences/*"):
        if refFolderName.endswith("_" + str(settings["general"]["log_counter"])):
            print(refFolderName)
            refSequenceFolders.append(refFolderName)
    
    with ZipFile("/home/pi/open-optical-gating/open_optical_gating/app/static/refs.zip", "w") as zipObj:
        for refSequenceFolder in refSequenceFolders:
            print(f"Current ref folder = {refSequenceFolder}")
            for dirPath, dirNames, fileNames in os.walk(refSequenceFolder):
                print(dirPath, dirNames, fileNames)
                for fileName in fileNames:
                    zipObj.write(
                        os.path.join(dirPath, fileName),
                        os.path.relpath(os.path.join(dirPath, fileName), os.path.join(refSequenceFolders[0], "../.."))
                        )
            
    return render_template("post_sync.html")

def confirm(host = "http://google.com"):
    """
    Function to confirm an internet connection.
    """
    i = 0
    max_i = 10000
    while True:
        i += 1
        if i > max_i:
            i = 0
        loadingString = "Waiting for an internet connection to be established..." + "."*(int(i/1000))  + " "*(int(max_i/1000) - int(i/1000))
        sys.stdout.write("\r" + loadingString)
        try:
            urllib.request.urlopen(host)
            return True
        except:
            continue
        
if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", threaded = True)

