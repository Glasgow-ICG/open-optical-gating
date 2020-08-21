import asyncio
import websockets
import sys, time
import numpy as np

from pixelarray import PixelArray
import sockets_comms as comms

async def send_frame():
    # Sends a frame message to the server, and expects the server to send a frame message back as response.
    # (Note that that is not the normal behaviour of websocket_optical_gater.py,
    # although it's what websockets_example_server.py does
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        t0 = time.time()
        arr = PixelArray(np.zeros((1,1)).astype('uint8'), metadata={"timestamp": 0})
        t1 = time.time()
        arrayMessage = comms.EncodeFrameMessage(arr)
        t2 = time.time()
        await websocket.send(arrayMessage)
        t3 = time.time()
        #print(f"> {arrayMessage}")
        
        responseRaw = await websocket.recv()
        t4 = time.time()

        if sys.argv[1] == "echo":
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
            print("Received response {0}".format(comms.DecodeMessage(responseRaw)))

asyncio.get_event_loop().run_until_complete(send_frame())
