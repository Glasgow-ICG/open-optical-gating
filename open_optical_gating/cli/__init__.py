from . import file_optical_gater
from . import websocket_optical_gater
from . import websocket_example_client
try:
    # Will fail except on RPi
    from . import pi_optical_gater
except:
    pass
