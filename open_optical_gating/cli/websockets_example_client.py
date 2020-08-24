import websockets, asyncio
import sys, time
import numpy as np
import json

from pixelarray import PixelArray
from file_optical_gater import FileOpticalGater
import sockets_comms as comms

async def send_frame(websocket, frame=None):
    # Sends a frame message to the server, and expects the server to send a frame message back as response.
    # (Note that that is not the normal behaviour of websocket_optical_gater.py,
    # although it's what websockets_example_server.py does
    t0 = time.time()
    if frame is None:
        frame = PixelArray(np.zeros((1,1)).astype('uint8'))
    else:
        frame = PixelArray(frame)
    if not "timestamp" in frame.metadata:
        frame.metadata["timestamp"] = time.time()
    t1 = time.time()
    arrayMessage = comms.EncodeFrameMessage(frame)
    t2 = time.time()
    await websocket.send(arrayMessage)
    t3 = time.time()

    responseRaw = await websocket.recv()
    t4 = time.time()

    if sys.argv[1] == "echo":
        # A test implementation that expects to be talking to a server behaving like websockets_example_server.py
        arrBack = comms.ParseFrameMessage(comms.DecodeMessage(responseRaw))
        t5 = time.time()

        print("Array creation {0:.3f}ms, json.dumps {1:.3f}ms, websocket.send {2:.3f}ms".format((t1-t0)*1e3, (t2-t1)*1e3, (t3-t2)*1e3))
        try:
            decT = arrBack.metadata['decodeTimes']
        except:
            decT = [0, 0]
        print("Received at server {0:.3f}ms after send started, decode {1:.3f}ms and server ready to use object".format((decT[0]-t2)*1e3,
                                                                                                                        (decT[1]-decT[0])*1e3))
        print("Server ready to use after total {0:.3f}ms from when we started sending".format((decT[1]-t1)*1e3))
        print("Response arrived {0:.3f}ms after send started".format((t4-t2)*1e3))
        print("Decode response {0:.3f}ms".format((t5-t4)*1e3))
        print("= Total round-trip {0:.3f}ms".format((t5-t1)*1e3))
    else:
        # A test implementation that expects to be talking to a real sync server like websocket_optical_gater.py
        response = comms.DecodeMessage(responseRaw)
        return response

async def send_test_frame(uri):
    async with websockets.connect(uri) as websocket:
        asyncio.get_event_loop().run_until_complete(send_frame(websocket))

async def send_from_file(uri, settings):
    """Emulated data capture for a set of sample brightfield frames."""
    async with websockets.connect(uri) as websocket:
        file_source_path=settings["path"]

        # We do instantiate a FileOpticalGater object, but actually all we use it for is to get frames from the file.
        # We don't then ask it to analyze those frames, we just send them on to the server.
        file_source = FileOpticalGater(file_source_path=settings["path"], settings=settings)
        while not file_source.stop:
            response = await send_frame(websocket, file_source.next_frame())
            print("Got sync response: ", response)

if __name__ == "__main__":
    uri = "ws://localhost:8765"
    if sys.argv[1] == "file":
        # Reads data from settings json file
        if len(sys.argv) > 2:
            settings_file = sys.argv[2]
        else:
            settings_file = "settings.json"
        
        with open(settings_file) as data_file:
            settings = json.load(data_file)
        
        asyncio.get_event_loop().run_until_complete(send_from_file(uri, settings))
    else:
        asyncio.get_event_loop().run_until_complete(send_test_frame(uri))
