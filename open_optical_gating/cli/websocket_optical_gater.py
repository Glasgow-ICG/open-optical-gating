"""Extension of CLI Open Optical Gating System for a remote client connecting over WebSockets"""

# Python imports
import sys
import json
import time

# Module imports
from loguru import logger
from skimage import io
import websockets, asyncio

# Local imports
import open_optical_gating.cli.optical_gater_server as server
from pixelarray import PixelArray
import sockets_comms as comms


class WebSocketOpticalGater(server.OpticalGater):
    """Extends the optical gater server for a remote client connecting over WebSockets
    """

    def __init__(self, settings=None, ref_frames=None, ref_frame_period=None):
        """Function inputs:
            settings      dict  Parameters affecting operation (see default_settings.json)
        """

        # Initialise parent
        super(WebSocketOpticalGater, self).__init__(
            settings=settings, ref_frames=ref_frames, ref_frame_period=ref_frame_period,
        )

        # JT TODO: this is very temporary - for now the superclass requires this to be defined.
        # We either need to get the client tell us the framerate, or remove the need
        # for this attribute entirely and deduce the framerate from individual frame timestamps.
        # The latter would be better, as long as the timestamps on individual frames are reliable.
        # We need to make sure they are, though, or our predictions will be off anyway!
        self.framerate = 80

    async def message_handler(self, websocket):
        # Wait for messages from the remote client
        async for rawMessage in websocket:
            message = comms.DecodeMessage(rawMessage)

            if not "type" in message:
                logger.critical(
                    "Ignoring unknown message with no 'type' specifier. Message was {0}".format(
                        message
                    )
                )
            elif message["type"] == "frame":
                # Do the synchronization analysis on the frame in this message
                pixelArrayObject = comms.ParseFrameMessage(message)
                if not "timestamp" in pixelArrayObject.metadata:
                    logger.critical(
                        "Received a frame that does not have compulsory metadata. We will ignore this frame."
                    )
                    continue
                logger.debug(
                    "Received frame with timestamp {0:.3f}".format(
                        pixelArrayObject.metadata["timestamp"]
                    )
                )
                if "sync" in pixelArrayObject.metadata:
                    logger.critical(
                        "Received a frame that already has 'sync' metadata. We will overwrite this!"
                    )
                pixelArrayObject.metadata["sync"] = dict()

                # JT TODO: for now I just hack self.width and self.height, but this should get fixed as part of the PixelArray refactor
                self.height, self.width = pixelArrayObject.shape
                self.analyze_pixelarray(pixelArrayObject)

                # Send back to the client the metadata we have added to the frame as part of the sync analysis.
                # This will include whether or not a trigger is predicted, and when.
                keys = ["optical_gating_state", "unwrapped_phase", "predicted_trigger_time_s", "trigger_type_sent"]
                response_dict = dict()
                for k in keys:
                    if k in pixelArrayObject.metadata:
                        response_dict[k] = pixelArrayObject.metadata[k]

                returnMessage = comms.EncodeFrameResponseMessage(response_dict)
                await websocket.send(returnMessage)
            else:
                logger.critical(
                    "Ignoring unknown message of type {0}".format(message["type"])
                )

    def run_server(self, host="localhost", port=8765):
        """ Blocking call that runs the WebSockets server, acting on client messages (mostly frames, probably)
            Function inputs:
              host          str   Host address to use for socket server
              port          int   Port to use for socket server
            """
        start_server = websockets.serve(
            lambda ws, p: self.message_handler(ws), "localhost", 8765
        )
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()


def run(settings):
    logger.success("Initialising gater...")
    analyser = WebSocketOpticalGater(settings=settings)
    logger.success("Running server...")
    analyser.run_server()


if __name__ == "__main__":
    t = time.time()
    # Reads data from settings json file
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = "settings.json"

    with open(settings_file) as data_file:
        settings = json.load(data_file)

    # Runs the server
    run(settings)
