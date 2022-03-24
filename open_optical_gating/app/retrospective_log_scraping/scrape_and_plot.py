import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Local imports
from retrospective_log_scraping.helper_functions import log_scraper as log_scraper
from retrospective_log_scraping.helper_functions import phase_interpolator as phase_interpolator
from retrospective_log_scraping.helper_functions import plot_maker as plot_maker

def run(logPath, keyPath):
    
    # Call the log_scraper function to generate the required dictionary
    scrapedDict = log_scraper.LogFileScraper(logPath, keyPath)
    
    # Interpolate for phase at trigger times and compute the estimated phase error
    triggerPhases, triggerPhaseErrors = phase_interpolator.TriggerPhaseInterpolation(
        scrapedDict["timeStamps"], 
        scrapedDict["states"],
        scrapedDict["phases"], 
        scrapedDict["triggerTimes"], 
        scrapedDict["targetPhases"]
    )
    plot_maker.plot_unwrapped_phase(
        scrapedDict["timeStamps"],
        scrapedDict["states"],
        scrapedDict["phases"]
    )
    plot_maker.plot_triggers(
        scrapedDict["timeStamps"],
        scrapedDict["states"],
        scrapedDict["phases"],
        scrapedDict["triggerTimes"],
        scrapedDict["targetPhases"],
        triggerPhases
    )
    plot_maker.plot_phase_histogram(
        np.asarray(triggerPhases), 
        scrapedDict["targetPhases"]
    )
    plot_maker.plot_phase_error_histogram(
        triggerPhaseErrors
    )
    plot_maker.plot_phase_error_with_time(
        scrapedDict["triggerTimes"], 
        triggerPhaseErrors
    )   
    plot_maker.plot_framerate(
        scrapedDict["timeStamps"],
        scrapedDict["states"]
    )  

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
