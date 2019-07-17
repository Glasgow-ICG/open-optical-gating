from image_loading import *
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.image as img
import plist_wrapper as jPlist

def PlotPhases(basePath, key='phase_from_offline_sync_analysis'):
    # Make a scatter plot of phase vs z to check how uniform the sampling is
    (images, dummy) = LoadAllImages(basePath, loadImageData = False)

    knownZ = []
    knownPhases = []
    for i in range(len(images)):
        if key in images[i].plist:
            knownZ.append(images[i].frameKeyPath('stage_positions.last_known_z'])
            knownPhases.append(images[i].frameKeyPath(key))
    knownZ = np.array(knownZ)
    knownPhases = np.array(knownPhases)

    plt.plot(knownPhases, knownZ, '.')
    plt.show()

PlotPhases('/Users/jonny/Movies/2017-01-11 PARTIAL/2017-01-11 14.54.34 vid tail piv/Brightfield subset', 'postacquisition_phase')
