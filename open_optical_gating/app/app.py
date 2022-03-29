#!/usr/bin/env python

"""Web app for interacting with the open optical gating system.
Much of this was inspired by https://github.com/miguelgrinberg/flask-video-streaming"""

# Imports
import sys, io, os, glob, shutil, cgi, json, time, cv2
import urllib.request
import numpy as np
from importlib import import_module
from multiprocessing import Process, Queue
from zipfile import ZipFile
from flask import Flask, render_template, Response, request, jsonify, session, stream_with_context

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
        shutter_speed_us = settings["brightfield"]["shutter_speed_us"]
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
    
@app.route("/print-data/")
def print_data():
    """Streams live sync readouts to the 'Sync' web-server page."""
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
        time_limit_seconds = settings["general"]["time_limit_seconds"], 
        trigger_update_number = settings["reference"]["update_after_n_criterions"],
        trigger_limit = settings["general"]["trigger_limit"]
    )

@app.route("/")
def index():
    """Setup for the live run page."""
    # Attempt to kill the camera object used by the settings pane.
    # Not currently working
    if (
        "camera" in globals()
        ):
        camera.stop = True
        camera.stop_now()
        os.system("sudo reboot")
        
    # Initialise global queues to send/recieved data from the concurrent sync process
    global eventQueue
    eventQueue = Queue()
    global stopQueue
    stopQueue = Queue()

    return render_template(
        "live.html", 
        time_limit_seconds = settings["general"]["time_limit_seconds"], 
        trigger_update_number = settings["reference"]["update_after_n_criterions"],
        trigger_limit = settings["general"]["trigger_limit"]
        )

@app.route("/update-setting-live/", methods = ["POST"])
def update_setting_live():
    """
    Update the relevant settings as prompted by the user via the web-server's 'Sync' pane.
    """
    # Get the requested change
    changeList = list(request.form.items())
    
    # Update the relevant setting in the settings dict
    if "limit" in changeList[0][0]:
        settings["general"][changeList[0][0]] = int(changeList[0][1])
    else:
        settings["reference"][changeList[0][0]] = int(changeList[0][1])
    
    # Dump the updated settings to json
    with open(settings_file, "w") as fp:
        json.dump(settings, fp, indent = 4)
        
    return index() 
    
@app.route("/start-live/")
def start_live():
    print("SYNC INITIATED...")
    
    # Initialise a sync object
    syncObject = sync.PiOpticalGater(settings = settings)
    
    # Setup the appropriate trigger pins
    syncObject.setup_pins()
    
    # Assign the start_sync function to a new process and start
    syncProcess = Process(target = syncObject.start_sync, args = (eventQueue, stopQueue, ))
    syncProcess.start()
    
    # A while loop to keep the main process active
    while stopQueue.empty():
        # Sleep to release GIL
        time.sleep(0.0001)
    
    # Long sleep to let all concurent processes and threads complete
    time.sleep(0.2)
    print("SHUTTING DOWN PI PROCESS...")
    
    # Terminate and join the syncProcess
    syncProcess.terminate()
    syncProcess.join()
    
    print("PI PROCESS SHUT DOWN AND SYNC COMPLETE...")
    return index()

@app.route("/stop-live/")
def stop_live():
    """Stop live sync."""
    # Place True in the global stopQueue
    stopQueue.put(True)
    
    # Sleep to wait for all processes/threads to complete before rendering the page
    time.sleep(1)
    return index()

@app.route("/post-sync-setup/")
def post_sync_setup():
    """A page to display relevant plots generated from the logs."""
    
    # Find the most recent log file in the user_log_folder
    all_logs = glob.glob("/home/pi/user_log_folder/*")
    most_recent_log = max(all_logs, key = os.path.getctime)
    
    # If the log is a real log (not empty) we move it to the static folder, overwriting the previous log
    if len(open(most_recent_log).readlines()) > 5:
        os.rename(most_recent_log, "/home/pi/open-optical-gating/open_optical_gating/app/static/most_recent_log.log")
        
    # Run the log scraper on this log file
    scrape.run("/home/pi/open-optical-gating/open_optical_gating/app/static/most_recent_log.log", '/home/pi/open-optical-gating/open_optical_gating/app/retrospective_log_scraping/log_keys.json')
    
    # Zip all generated plots in the static folder to be downloaded by the user
    with ZipFile( "/home/pi/open-optical-gating/open_optical_gating/app/static/plots.zip", "w") as zipObj:
        for plotPath in glob.glob("/home/pi/open-optical-gating/open_optical_gating/app/static/*.jpg"):
            zipObj.write(plotPath, os.path.basename(plotPath))
    return render_template("post_sync.html")

def confirm(host = "http://google.com"):
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
    if confirm():
        print("\nConnection established; running script...")
        app.run(debug = True, host = "0.0.0.0", threaded = True)
    else:
        print("\nInternet connection could not be established...")
        time.sleep(10)
