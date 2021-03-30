import websockets, asyncio
import sys, time, os
import numpy as np
import json
import matplotlib.pyplot as plt
from loguru import logger

from . import pixelarray
from . import file_optical_gater
from . import sockets_comms as comms


async def send_frame(websocket, frame=None, expect_echo=False):
    # Sends a frame message to the server, and expects the server to send a frame message back as response.
    # (Note that that is not the normal behaviour of websocket_optical_gater.py,
    # although it's what websockets_example_server.py does
    t0 = time.time()
    if frame is None:
        # Create a dummy frame of zeroes, just for test purposes
        frame = pixelarray.PixelArray(np.zeros((300, 300)).astype("uint8"))
    else:
        frame = pixelarray.PixelArray(frame)
    if not "timestamp" in frame.metadata:
        frame.metadata["timestamp"] = time.time()
    t1 = time.time()
    arrayMessage = comms.EncodeFrameMessage(frame)
    t2 = time.time()
    await websocket.send(arrayMessage)
    t3 = time.time()

    responseRaw = await websocket.recv()
    t4 = time.time()

    if expect_echo:
        # A test implementation that expects to be talking to a server behaving like websockets_example_server.py
        arrBack = comms.ParseFrameMessage(comms.DecodeMessage(responseRaw))
        t5 = time.time()

        print(
            "Array creation {0:.3f}ms, json.dumps {1:.3f}ms, websocket.send {2:.3f}ms".format(
                (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
            )
        )
        try:
            decT = arrBack.metadata["decodeTimes"]
        except:
            decT = [0, 0]
        print(
            "Received at server {0:.3f}ms after send started, decode {1:.3f}ms and server ready to use object".format(
                (decT[0] - t2) * 1e3, (decT[1] - decT[0]) * 1e3
            )
        )
        print(
            "Server ready to use after total {0:.3f}ms from when we started sending".format(
                (decT[1] - t1) * 1e3
            )
        )
        print("Response arrived {0:.3f}ms after send started".format((t4 - t2) * 1e3))
        print("Decode response {0:.3f}ms".format((t5 - t4) * 1e3))
        print("= Total round-trip {0:.3f}ms".format((t5 - t1) * 1e3))
    else:
        # A test implementation that expects to be talking to a real sync server like websocket_optical_gater.py
        response = comms.DecodeMessage(responseRaw)
        return response, frame.metadata["timestamp"]


async def send_test_frame(uri, expect_echo=False):
    async with websockets.connect(uri) as websocket:
        response, timestamp = await send_frame(websocket, expect_echo=expect_echo)


async def send_from_file(uri, settings, expect_echo=False):
    """Emulated data capture for a set of sample brightfield frames."""
    async with websockets.connect(uri) as websocket:
        source = settings["input_tiff_path"]

        # We do instantiate a FileOpticalGater object, but actually all we use it for is to get frames from the file.
        # We don't then ask it to analyze those frames, we just send them on to the server.
        file_source = file_optical_gater.FileOpticalGater(source=settings["input_tiff_path"], settings=settings)
        phases = []
        times = []
        sent_trigger_times = []
        while not file_source.stop:
            frame = file_source.next_frame(force_framerate=False)
            response, timestamp = await send_frame(
                websocket, frame, expect_echo
            )
            print("Got sync response: ", response)
            if "unwrapped_phase" in response["sync"]:
                phases.append(response["sync"]["unwrapped_phase"] % (2 * np.pi))
                times.append(timestamp)
            if ("trigger_type_sent" in response["sync"]) and (response["sync"]["trigger_type_sent"] > 0):
                print('prediction', response["sync"]["predicted_trigger_time_s"])
                sent_trigger_times.append(response["sync"]["predicted_trigger_time_s"])

        plt.figure()
        plt.title("Zebrafish heart phase with trigger fires")
        plt.plot(times,
                 phases,
                 label="Heart phase"
                 )
        # JT TODO: instead of the second '0' in np.full, below, we should be using the target sync phase (but we don't know that).
        # Note that the file_optical_gater also has a (different) bug here where it does not allow for the fact that this changes with LTU!
        plt.scatter(
                    np.array(sent_trigger_times),
                    np.full(
                            max(len(sent_trigger_times), 0), 0,
                            ),
                    10,
                    color="r",
                    label="Trigger fire",
                    )
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (rad)")
        plt.show()

def run(args, desc):
    '''
        Run a websocket-based optical gater server, configured based on the .json file provided
        
        Params:   raw_args   list    Caller should normally pass sys.argv here
        desc       str     Description to provide as command line help description
        '''
    def add_extra_args(parser):
        parser.add_argument("-l", "--loopback", dest="loopback", action="store_true", help="expecting to interact with a loopback server")
        parser.add_argument("-t", "--test-frame", dest="test_frame", action="store_true", help="just send a single test frame to the server")
        parser.add_argument("-u", "--server-uri", dest="uri", default="ws://localhost:8765", help="URI for server to connect to")
    settings = file_optical_gater.load_settings(args, desc, add_extra_args)
    expect_echo = settings["parsed_args"].loopback

    if settings["parsed_args"].test_frame:
        asyncio.get_event_loop().run_until_complete(send_test_frame(settings["parsed_args"].uri, expect_echo))
    else:
        asyncio.get_event_loop().run_until_complete(send_from_file(settings["parsed_args"].uri, settings, expect_echo))

if __name__ == "__main__":
    run(sys.argv[1:], "Run a websocket-based optical gater client, sending image data to a separate server instance for processing")
