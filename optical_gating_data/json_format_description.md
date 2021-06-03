### Generic settings
- "brightfield_framerate" (int): For pi_optical_gater and file_optical_gater, this controls the actual camera framerate. For other classes this must match the framerate configured on the camera, since we need to know that information for the optical gating predictions. 
- "prediction_latency_s" (float): Expected minimum possible round-trip time (in seconds) from the time on the frame timestamp to the time when a trigger could be sent when requested by our gating code. Needs to allow time for frame transfer, gating analysis, return communications, and configuring of electrical output signal. This should ideally be measured under live conditions, but is typically of the order of 10-20ms.
- "update_after_n_triggers" (int, optional): Automatically acquire a new sequence of reference frames after this many triggers have been generated. 
- "frame_buffer_length" (int): Maximum number of recent frames to retain as part of the sync analysis. Should be enough frames to cover at least one heartbeat; can be longer for debug purposes.  
- "min_heart_rate_hz" (float): Minimum heart rate we are expecting. This is used while establishing a reference heartbeat, and it caps the amount of computational work we will do. This is especially important to maintain performance in cases where a clear heartbeat is not visible in the provided images.
- "reference_sequence_dir" (str, optional): Path to directory where a TIFF will be saved containing the reference heartbeat images.
- "save_first_n_frames" (int, optional): The first N frames received will be saved to disk in the same way that reference sequences are. This may be useful for debug purposes.

### File optical gater specific settings
- "input_tiff_path" (str): Path to TIFF file containing a sequence of brightfield images to analyze. If this is a relative path, it will be treated as relative to the settings.json file (*not* the working directory from which the code has been run)
- "source_url" (str, optional): URL from which file can be downloaded if not already present on disk

### Raspberry Pi specific settings
- "brightfield_resolution" (int): image dimension (both width and height) to use on the Pi camera
- "shutter_speed_us" (int): brightfield camera exposure time (microseconds)
- "awb_mode": ???
- "exposure_mode": ???
- "image_denoise": ???
- "laser_trigger_pin": ???
- "fluorescence_camera_pins": dictionary containing:
    - "trigger" (int): ??
    - "SYNC-A" (int): ??
    - "SYNC-B" (int): ??
- "fluorescence_trigger_mode": ???
- "fluorescence_exposure_us" (int): ???

### Chas, these seem to be unused?
"live": 1,
"log": 0,
"frame_num": 0,
"analyse_time" (int): 1000,



