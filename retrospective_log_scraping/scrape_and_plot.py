import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Local imports
from helper_functions import log_scraper as log_scraper
from helper_functions import phase_interpolator as phase_interpolator
from helper_functions import plot_maker as plot_maker

def run(logPath, keyPath):

    startTime = time()
    
    # Call the log_scraper function to generate the required dictionary
    scrapedDict = log_scraper.LogFileScraper(logPath, keyPath)
    # Interpolate for phase at trigger times
    phases, phaseErrors = phase_interpolator.TriggerPhaseInterpolation(
        scrapedDict['timeStamps'], 
        scrapedDict['phases'], 
        scrapedDict['triggerTimes'], 
        scrapedDict['targetPhases']
    )
    print('Retrospective analysis performed in {0} seconds... \nPlotting graphs...'.format(np.round(time() - startTime, 5)))
    
    plot_maker.plot_triggers(
        scrapedDict["timeStamps"],
        scrapedDict["phases"],
        scrapedDict["triggerTimes"],
        scrapedDict["targetPhases"]
    )

    plot_maker.plot_phase_histogram(
        np.asarray(phases), 
        scrapedDict['targetPhases']
    )

    plot_maker.plot_phase_error_histogram(
        phaseErrors
    )

    plot_maker.plot_phase_error_with_time(
        scrapedDict['triggerTimes'], 
        phaseErrors
    )
    
if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])