## Imports
import numpy as np

from pprint import pprint
import time
import math
from copy import copy

def matchFrames(seq1,seq2,drift):
    # user must provide drift (see period.txt files)
    dx = drift[0]
    dy = drift[1]

    #apply shifts
    rectF = [0,seq1[0].shape[0],0,seq1[0].shape[1]]#X1,X2,Y1,Y2
    rect = [0,seq2[0].shape[0],0,seq2[0].shape[1]]#X1,X2,Y1,Y2

    if dx<=0:
        rectF[0] = -dx
        rect[1] = rect[1]+dx
    else:
        rectF[1] = rectF[1]-dx
        rect[0] = dx
    if dy<=0:
        rectF[2] = -dy
        rect[3] = rect[3]+dy
    else:
        rectF[3] = rectF[3]-dy
        rect[2] = +dy

    seq1 = seq1[:,rectF[0]:rectF[1],rectF[2]:rectF[3]]
    seq2 = seq2[:,rect[0]:rect[1],rect[2]:rect[3]]

    return seq1,seq2
