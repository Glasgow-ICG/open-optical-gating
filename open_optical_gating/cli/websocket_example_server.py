import asyncio
import websockets
import time

from pixelarray import PixelArray
from . import sockets_comms as comms

async def hello(websocket, path):
    frameMessage = await websocket.recv()
    t1 = time.time()
    arrayObject = comms.ParseFrameMessage(comms.DecodeMessage(frameMessage))
    t2 = time.time()
    
    arrayObject.metadata['decodeTimes'] = [t1, t2]

    returnMessage = comms.EncodeFrameMessage(arrayObject)
    await websocket.send(returnMessage)

start_server = websockets.serve(hello, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
