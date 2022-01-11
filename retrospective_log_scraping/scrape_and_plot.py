import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Local imports
from helper_functions import log_scraper as log_scraper
from helper_functions import phase_interpolator as phase_interpolator
from helper_functions import plot_maker as plot_maker

def run(logPath, keyPath):
    
    # Call the log_scraper function to generate the required dictionary
    scrapedDict = log_scraper.LogFileScraper(logPath, keyPath)
    # Interpolate for phase at trigger times and compute the estimated phase error
    phases, phaseErrors = phase_interpolator.TriggerPhaseInterpolation(
        scrapedDict["timeStamps"], 
        scrapedDict["states"],
        scrapedDict["phases"], 
        scrapedDict["triggerTimes"], 
        scrapedDict["targetPhases"]
    )
    
    plot_maker.plot_triggers(
        scrapedDict["timeStamps"],
        scrapedDict["states"],
        scrapedDict["phases"],
        scrapedDict["triggerTimes"],
        scrapedDict["targetPhases"]
    )

    plot_maker.plot_phase_histogram(
        np.asarray(phases), 
        scrapedDict["targetPhases"]
    )

    plot_maker.plot_phase_error_histogram(
        phaseErrors
    )

    plot_maker.plot_phase_error_with_time(
        scrapedDict["triggerTimes"], 
        phaseErrors
    )
    
    plot_maker.plot_framerate(
        scrapedDict["timeStamps"],
        scrapedDict["states"]
    )
    
    return scrapedDict
    
if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])