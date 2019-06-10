import j_py_sad_correlation as jps
import numpy as np
from numpy.lib.stride_tricks import as_strided
import time
import sys

# Generate two arrays containing random integers
a2 = np.round(np.random.randint(0, 255, (250,250))).astype('uint8')
b2 = np.round(np.random.randint(0, 255, (40,250,250))).astype('uint8')

diffs_using_c_code = jps.sad_with_references(a2, b2);
print(diffs_using_c_code)
diffs_using_python_code = np.sum(np.sum(np.abs(a2.astype('float')-b2.astype('float')), axis=2), axis=1)
print(diffs_using_python_code)
print ("success if these values are all zero:", (diffs_using_python_code - diffs_using_c_code).max(), (diffs_using_python_code - diffs_using_c_code).min())

# Generate two arrays containing random integers
a2 = np.round(np.random.randint(0, 255, (50,250,250))).astype('uint8')
b2 = np.round(np.random.randint(0, 255, (40,250,250))).astype('uint8')

diffs_using_c_code = jps.sad_grid(a2, b2);
print(diffs_using_c_code)
diffs_using_python_code = np.zeros((len(b2),len(a2)),dtype='float')
for i in range(len(a2)):
    for j in range(len(b2)):
        diffs_using_python_code[j,i] = np.sum(np.abs(a2[i].astype('float')-b2[j].astype('float')))
print(diffs_using_python_code)
print ("success if these values are all zero:", (diffs_using_python_code - diffs_using_c_code).max(), (diffs_using_python_code - diffs_using_c_code).min())
