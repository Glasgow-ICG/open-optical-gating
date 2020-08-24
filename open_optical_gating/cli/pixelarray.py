"""PixelArray class encapsulating a 2D camera image and its accompanying metadata"""

import numpy as np
import time
from pybase64 import b64encode, b64decode


class PixelArray(np.ndarray):
    # See explanations at https://numpy.org/doc/stable/user/basics.subclassing.html
    def __new__(cls, input_array, metadata=dict()):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', dict())

    def for_cbor(self):
        # Returns a representation of the array that is suitable for CBOR serialisation.
        return [self.shape, str(self.dtype), self.tobytes(), self.metadata]

    def for_json(self):
        # Returns a representation of the array that is suitable for JSON serialisation.
        # Note that we do not do this by subclassing JSONEncoder, because fast json encoders
        # such as orjson do not appear to support this mechanism, as far as I can see.
        arrayData = b64encode(self.tobytes()).decode()
        return [self.shape, str(self.dtype), arrayData, self.metadata]

def ArrayCBORDecode(jsonEncoded):
    arr = np.frombuffer(jsonEncoded[2], dtype=jsonEncoded[1]).reshape(jsonEncoded[0])
    result = PixelArray(arr)
    result.metadata = jsonEncoded[3]
    return result

def ArrayJSONDecode(jsonEncoded):
    arrayBytes = b64decode(jsonEncoded[2])
    arr = np.frombuffer(arrayBytes, dtype=jsonEncoded[1]).reshape(jsonEncoded[0])
    result = PixelArray(arr)
    result.metadata = jsonEncoded[3]
    return result

def MetadataArrayFromPixelArrayList(pixelArrayList, metadataKey):
    """ Given a list of PixelArray objects, returns a numpy array containing the
        value associated with the metadata entry for 'metadataKey' (e.g. 'timestamp')
        for each of the objects in the list. i.e. return an array containing the
        timestamp for each array object.
        
        Function inputs:
         pixelArrayList   list     List of PixelArray objects
         metadataKey      str      Metadata key to extract.
                                    Must be present for all objects in pixelArrayList.
        """
    return np.array([i.metadata[metadataKey] for i in pixelArrayList])
