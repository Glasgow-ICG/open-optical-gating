# My utility functions for reading and writing plist files.
# plistlib basically does what I want, but its interface has changed
# between python 2 and python 3 (which is annoying!), and so I have
# written a wrapper around it that works with either.
# Incidentally, plistlib seems pretty disappointingly slow at
# reading and parsing plists, but I don't think it's worth the bother to
# write my own, faster implementation.
import os, sys, time, warnings, plistlib

def readPlist(plistPath):
    if (sys.version_info > (3, 0)):
        # Python 3
        with open(plistPath, 'rb') as fp:
            pl = plistlib.load(fp)
        return pl
    else:
        # Python 2
        return plistlib.readPlist(plistPath)

def writePlist(rootObject, plistPath):
    if (sys.version_info > (3, 0)):
        # Python 3
        with open(plistPath, 'wb') as fp:
            plistlib.dump(rootObject, fp)
    else:
        # Python 2
        plistlib.writePlist(rootObject, plistPath)
