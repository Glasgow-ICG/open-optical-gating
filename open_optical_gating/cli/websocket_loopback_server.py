# Simple websocket server just sends back any frame message it receives,
# along with information about the time taken to decode it
import asyncio
import websockets
import time

from . import sockets_comms as comms

async def loopback(websocket, path):
    frameMessage = await websocket.recv()
    t1 = time.time()
    arrayObject = comms.ParseFrameMessage(comms.DecodeMessage(frameMessage))
    t2 = time.time()
    
    arrayObject.metadata['decodeTimes'] = [t1, t2]

    returnMessage = comms.EncodeFrameMessage(arrayObject)
    await websocket.send(returnMessage)

start_server = websockets.serve(loopback, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
