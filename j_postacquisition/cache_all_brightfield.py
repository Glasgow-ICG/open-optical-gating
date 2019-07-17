from analyze_dataset import *
import re
import shutil

# Scan the specified directory, loading (and caching) the plists from the brightfield folder in each stack directory.
# This is useful as a quick way of pulling the metadata from Edinburgh

# Takes two command-line arguments:
# 1. Parent directory. We scan within it for folders named Stack_%04d.
# 2. Cache directory. We will accumulate .npy files in this directory
#
# For each stack folder, we load the brightfield folder (and generate a cache file).
assert(len(sys.argv) == 3)
parentDirectory = sys.argv[1]
destCacheDir = sys.argv[2]

for file in os.listdir(parentDirectory):
    # See if filename has the correct form
    if re.match("^Stack [0-9]{4}$", file):
        directory = "%s/%s/Brightfield - Prosilica" % (parentDirectory, file)
        (images, estimatedPeriod) = LoadAllImages(directory, loadImageData=False)
        # Identify the cache file. This is a bit fragile, since it assumes there is only one cache file there, the one we just created.
        for file2 in os.listdir(directory):
            if re.match(".*npy$", file2):
                # Copy the cache file to a designated directory
                # We drop the hash and give it a simple name to make it clear which stack folder it refers to
                shutil.copy2('%s/%s' % (directory, file2), '%s/%s.npy' % (destCacheDir, file))
