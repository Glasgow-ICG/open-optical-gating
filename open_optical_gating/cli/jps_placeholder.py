"""Placeholder functions to replace j_py_sad_correlation,
    until I can get it compiling under Windows """

import numpy as np

def sad_with_references(frame, pastFrames):
    return np.sum(np.sum(np.abs(np.asarray(frame).astype('int')-np.asarray(pastFrames).astype('int')), axis=2), axis=1)

def sad_correlation(a, b):
    a = np.asarray(a).astype('int')   # We need to convert to int to avoid encountering overflow issues
    b = np.asarray(b).astype('int')
    sad_using_python_code = np.zeros((b.shape[0] - a.shape[0] + 1, b.shape[1] - a.shape[1] + 1))
    for y in range(sad_using_python_code.shape[1]):
        for x in range(sad_using_python_code.shape[0]):
            sad_using_python_code[x,y] = sum(sum(abs(a - b[x:x+a.shape[0], y:y+a.shape[1]])))
    return sad_using_python_code

def ssd_correlation(a, b):
    a = np.asarray(a).astype('int')   # We need to convert to int to avoid encountering overflow issues
    b = np.asarray(b).astype('int')
    sad_using_python_code = np.zeros((b.shape[0] - a.shape[0] + 1, b.shape[1] - a.shape[1] + 1))
    for y in range(sad_using_python_code.shape[1]):
        for x in range(sad_using_python_code.shape[0]):
            sad_using_python_code[x,y] = sum(sum((a - b[x:x+a.shape[0], y:y+a.shape[1]])**2))
    return sad_using_python_code
