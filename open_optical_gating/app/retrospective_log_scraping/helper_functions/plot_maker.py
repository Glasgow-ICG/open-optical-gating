import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle

dpi = 400
figsize = (4,3)
lineWidth = 1.2
markerSize = 5
plt.switch_backend('pdf')

def plot_unwrapped_phase(timeStamps, states, phases):
    """Plot unwrapped phase."""
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Zebrafish unwrapped phase")
    plt.plot(
        timeStamps[states == 'sync'],
        phases,
        color = "tab:green",
        linewidth = lineWidth, 
        zorder = 5
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Unwrapped phase (radians)")
    plt.tight_layout()
    plt.savefig("/home/pi/open-optical-gating/open_optical_gating/app/static/unwrappedPhase.jpg", dpi = dpi)
    plt.close()

def plot_triggers(timeStamps, states, phases, triggerTimes, targetPhases, triggerPhases):
    """
    Plot the phase vs. time sawtooth line with trigger events.
    Note: The very first target phase is used to denote the target phase of the entire 
    sync. This is due to the fact that the actual target phase is decided with respect
    to the initial reference sequence.
    """
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Zebrafish heart phase with trigger fires")
    plt.plot(
        timeStamps[states == 'sync'],
        phases % (2 * np.pi),
        label= "Heart phase",
        color =  "tab:green",
        linewidth = lineWidth,
        zorder = 5
    )
    plt.scatter(
        triggerTimes, 
        np.asarray(triggerPhases) % (2 * np.pi),
        s = markerSize,
        label = "Real",
        color = "tab:blue",
        zorder = 15
    )
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.tight_layout()
    plt.savefig("/home/pi/open-optical-gating/open_optical_gating/app/static/wrappedPhase.jpg", dpi = dpi)
    plt.close()

def plot_phase_histogram(triggerPhases, targetPhases):
    """Plots a histogram depicting the frequency of occurance of the phase at each sent trigger time. """
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Frequency density of triggered phase")
    plt.hist(
        triggerPhases % (2 * np.pi), 
        bins = np.arange(0, 2 * np.pi, 0.01), 
        color = "tab:green", 
        label = "Triggered phase"
        )
    x_1, x_2, y_1, y_2 = plt.axis()

    uniqueTargetPhases = np.unique(targetPhases)
    for ph in uniqueTargetPhases:
        plt.plot(np.full(2, ph), (y_1, y_2),"red", label = "Target phase",)
        
    plt.xlabel("Triggered phase (rad)")
    plt.ylabel("Frequency")
    plt.axis((x_1, x_2, y_1, y_2))
    plt.tight_layout()
    plt.savefig("/home/pi/open-optical-gating/open_optical_gating/app/static/phaseHistogram.jpg", dpi = dpi)
    plt.close()

def plot_phase_error_histogram(triggerPhaseErrors):
    """Plots a histogram representing the frequency estimated phase errors."""
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Trigger phase error")
    plt.hist(
        triggerPhaseErrors,
        bins = np.arange(np.min(triggerPhaseErrors), np.max(triggerPhaseErrors) + 0.1,  0.03), 
        color = "tab:green", 
        label = "Triggered phase"
    )
    x_1, x_2, y_1, y_2 = plt.axis()
    plt.xlabel("Phase error (rad)")
    plt.ylabel("Frequency")
    plt.axis((x_1, x_2, y_1, y_2))
    plt.tight_layout()
    plt.savefig("/home/pi/open-optical-gating/open_optical_gating/app/static/phaseErrorHistogram.jpg", dpi = dpi)
    plt.close()

def plot_phase_error_with_time(triggerTimes, triggerPhaseErrors):
    """Plots the estimated phase error associated with each sent trigger over time."""
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title('Trigger phase error with time')
    plt.scatter(
        triggerTimes, 
        triggerPhaseErrors, 
        color = 'tab:green',
        s = markerSize
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Phase error (rad)')
    plt.ylim(-np.pi, np.pi)
    plt.tight_layout()
    plt.savefig("/home/pi/open-optical-gating/open_optical_gating/app/static/phaseErrorWithTime.jpg", dpi = dpi)
    plt.close()
    
def plot_framerate(timeStamps, states, windowSize = 10):
    """
    Plots a scatter of the 10-frame-averaged framerate while highlighting time periods in which the software
    was synchronising or updating the reference sequence.
    """
    def frame_rate_calculator(timeStamps, windowSize):
        timeDifferences = np.asarray([timeStamps[i] - timeStamps[i - 1] for i in range(1, len(timeStamps))])
        framerates = [1/np.mean(timeDifferences[i:(i + windowSize)]) for i in range(len(timeDifferences) - windowSize)]
        return framerates
    
    # Compute framerate for sync and non-sync states
    syncFramerates = frame_rate_calculator(timeStamps[states == 'sync'], windowSize)
    nonSyncFramerates = frame_rate_calculator(timeStamps[states != 'sync'], windowSize)
    
    # Find the timestamps at which the state changed (e.g from 'determine' to 'sync')
    changeTimes = [timeStamps[i] for i in range(1, len(states)) if states[i] != states[i - 1]] 
    
    # If the script is terminated before a sync-run is complete, we add the final timestamp as the end point of the sync 
    changeTimes.append(timeStamps[-1]) if len(changeTimes) % 2 != 0 else None
    changeTimes = np.asarray(changeTimes)
    startSyncTimes, endSyncTimes = changeTimes[::2], changeTimes[1::2]

    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Framerate")
    plt.scatter(
        timeStamps[states == 'sync'][(windowSize + 1):], 
        syncFramerates,
        color = 'tab:green',
        s = markerSize,
        label = "Sync framerate"
    )
    plt.scatter(
        timeStamps[states != 'sync'][(windowSize + 1):], 
        nonSyncFramerates,
        color = 'tab:red',
        s = markerSize,
        label = "Update framerate"
    )
    plt.bar(
        x =  (startSyncTimes + endSyncTimes)/2, 
        height = np.max(syncFramerates), 
        width = (endSyncTimes - startSyncTimes), 
        color = 'tab:green', 
        alpha = 0.2,
        label = 'Sync state'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Framerate (fps)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/pi/open-optical-gating/open_optical_gating/app/static/framerate.jpg", dpi = dpi)
    plt.close()
    
def plot_temperature_with_time(timeStamps, temperatures):
    """Plot raspberry pi temperature."""
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Pi temperature with time")
    plt.plot(
        timeStamps, 
        temperatures[1:], 
        linewidth = lineWidth,
        color = "tab:green"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (Celcius)")
    plt.tight_layout()
    plt.savefig("/home/pi/open-optical-gating/open_optical_gating/app/static/temperature.jpg", dpi = dpi)
    plt.close()
