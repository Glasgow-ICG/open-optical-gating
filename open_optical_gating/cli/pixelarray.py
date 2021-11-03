"""PixelArray class encapsulating a 2D camera image and its accompanying metadata.

Frame metadata:
    Dictionary keys that are used:
        "timestamp"                 Timestamp associated with the frame.
                                    This can be in any timebase (e.g. computer, camera hardware, ...), and predicted future trigger times will be computed and returned
                                    in that same timebase.
        "unwrapped_phase"           Unwrapped phase based on prospective optical gating phase matching.
        "sad_min"                   Position of minimum SAD with reference period.
        "predicted_trigger_time_s"  The predicted trigger as determined by prospective optical gating
        "trigger_type_sent"         Trigger type sent; 0 is no trigger; 1 and 2 are a sent trigger
        "processing_rate_fps"       Current frame processing rate in frames per second
"""

import numpy as np
import time
from pybase64 import b64encode, b64decode


class PixelArray(np.ndarray):
    # See explanations at https://numpy.org/doc/stable/user/basics.subclassing.html
    def __new__(cls, input_array, metadata=dict()):
        # Input array is an already formed ndarray (or subclass) instance
        # We first cast to be our class type, and force the array to be contiguous
        obj = np.ascontiguousarray(input_array).view(cls)

        # TODO: I would like to do the following test here:
        # if (isinstance(input_array, PixelArray)):
        # ... but we end up comparing <class 'open_optical_gating.cli.pixelarray.PixelArray'>
        #     with <class 'pixelarray.PixelArray'>, which is not the same.
        #     We need to work out how to deal with that issue, but for now I can hack it
        #     by testing for the case where input_array is NOT of the exact type np.ndarray
        if not type(input_array) == np.ndarray:
            if len(metadata) != 0:
                raise TypeError(
                    "Cannot provide out-of-line metadata when constructing from an existing PixelArray object (with its own metadata)"
                )
            obj.metadata = input_array.metadata.copy()
        else:
            obj.metadata = metadata.copy()
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, "metadata", dict())

    def for_cbor(self):
        # Returns a representation of the array that is suitable for CBOR serialisation.
        return { "shape": self.shape,
                 "dtype": str(self.dtype),
                 "pixels": self.tobytes(),
                 "metadata": self.metadata }

    def for_json(self):
        # Returns a representation of the array that is suitable for JSON serialisation.
        # Note that we do not do this by subclassing JSONEncoder, because fast json encoders
        # such as orjson do not appear to support this mechanism, as far as I can see.
        arrayData = b64encode(self.tobytes()).decode()
        return { "shape": self.shape,
                 "dtype": str(self.dtype),
                 "pixels": arrayData,
                 "metadata": self.metadata }


def ArrayCBORDecode(arrayDict):
    arr = np.frombuffer(arrayDict["pixels"], dtype=arrayDict["dtype"]).reshape(arrayDict["shape"])
    result = PixelArray(arr)
    result.metadata = arrayDict["metadata"]
    return result


def ArrayJSONDecode(arrayDict):
    arrayBytes = b64decode(arrayDict["pixels"])
    arr = np.frombuffer(arrayBytes, dtype=arrayDict["dtype"]).reshape(arrayDict["shape"])
    result = PixelArray(arr)
    result.metadata = arrayDict["metadata"]
    return result


def get_metadata_from_list(pixelArrayList, metadataKey, onlyIfKeyPresent=None):
    """ Given a list of PixelArray objects, returns a numpy array containing the
        value associated with the metadata entry for 'metadataKey' (e.g. 'timestamp')
        for each of the objects in the list. i.e. return an array containing the
        timestamp for each array object.
        
        Function inputs:
         pixelArrayList   list          List of PixelArray objects
         metadataKey      str or list   Metadata key to extract.
                                        By default, this must be present for all objects in pixelArrayList.
                                        If a list, returns an array with n columns.
         onlyIfKeyPresent str or None   If provided, we only consider entries in pixelArrayList that contain this key/value pair
                                        This is useful e.g. for selecting out only frames where a trigger was sent
        """
    if isinstance(metadataKey, str):
        if onlyIfKeyPresent is not None:
            return np.array([i.metadata[metadataKey] for i in pixelArrayList if onlyIfKeyPresent in i.metadata])
        else:
            return np.array([i.metadata[metadataKey] for i in pixelArrayList])
    elif isinstance(metadataKey, list):
        if onlyIfKeyPresent is not None:
            raise ValueError('onlyIfKeyPresent is not supported in combination with key lists')
        return np.array(
            [[i.metadata[m] for i in pixelArrayList] for m in metadataKey]
        ).T
    else:
        raise TypeError(
            'Parameter metadataKey must be of type "str" or "list" (but is type "{0}")'.format(
                type(metadataKey)
            )
        )
