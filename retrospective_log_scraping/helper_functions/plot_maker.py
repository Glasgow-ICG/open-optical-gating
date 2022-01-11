import numpy as np
import matplotlib.pyplot as plt

dpi = 200
figsize = (5,3)
lineWidth = 1.2
markerSize = 5

def plot_triggers(timeStamps, states, phases, triggerTimes, targetPhases):
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
        np.full(max(len(triggerTimes), 0), targetPhases[0]),
        color= "tab:red",
        label= "Trigger fire",
        s = markerSize,
        zorder = 10
    )
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.tight_layout()
    plt.show()

def plot_phase_histogram(phases, targetPhases):
    """
    Plots a histogram depicting the frequency of occurance of the phase at each sent trigger time. 
    """
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Frequency density of triggered phase")
    plt.hist(
        phases % (2 * np.pi), 
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
    plt.show()

def plot_phase_error_histogram(phaseErrors):
    """
    Plots a histogram representing the frequency estimated phase errors.
    """
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title("Frequency density of phase error at trigger")
    plt.hist(
        phaseErrors,
        bins = np.arange(np.min(phaseErrors), np.max(phaseErrors) + 0.1,  0.03), 
        color = "tab:green", 
        label = "Triggered phase"
    )
    x_1, x_2, y_1, y_2 = plt.axis()
    plt.xlabel("Phase error (rad)")
    plt.ylabel("Frequency")
    plt.axis((x_1, x_2, y_1, y_2))
    plt.tight_layout()
    plt.show()

def plot_phase_error_with_time(triggerTimes, phaseErrors):
    """
    Plots the estimated phase error associated with each sent trigger over time.
    """
    plt.figure(figsize = figsize, dpi = dpi)
    plt.title('Triggered phase error with time')
    plt.scatter(
        triggerTimes, 
        phaseErrors, 
        color = 'tab:green',
        s = markerSize
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Phase error (rad)')
    plt.ylim(-np.pi, np.pi)
    plt.tight_layout()
    plt.show()
    
def plot_framerate(timeStamps, states, windowSize = 10):
    """
    Plots a scatter of the 10-frame-averaged framerate while highlighting time periods in which the software
    was synchronising or updating the reference sequence.
    """
    
    def frame_rate_calculator(timeStamps, windowSize):
        # Compute the difference between frame timestamps
        timeDifferences = np.asarray([timeStamps[i] - timeStamps[i - 1] for i in range(1, len(timeStamps))])
        # Compute the framerate averaged over the provided 'windowSize'
        framerates = [1/np.mean(timeDifferences[i:(i + windowSize)]) for i in range(len(timeDifferences) - windowSize)]
        return framerates
    
    # Compute framerate for sync and non-sync states
    syncFramerates = frame_rate_calculator(timeStamps[states == 'sync'], windowSize)
    nonSyncFramerates = frame_rate_calculator(timeStamps[states != 'sync'], windowSize)
    
    # Finding the timestamps at which the state changed (e.g from 'determine' to 'sync')
    changeTimes = [timeStamps[i] for i in range(1, len(states)) if states[i] != states[i - 1]] 
    
    # If the script is terminated before a sync-run is complete, we must add the final timestamp as the end point of the sync 
    changeTimes.append(timeStamps[-1]) if len(changeTimes) % 2 != 0 else None
    changeTimes = np.asarray(changeTimes)
    
    # Form parameters for translucent bar graph depicting when the system was in the 'sync state'
    startSyncTimes, endSyncTimes = changeTimes[::2], changeTimes[1::2]
    barCentres = (startSyncTimes + endSyncTimes)/2
    barWidths = (endSyncTimes - startSyncTimes)
        
    plt.figure(figsize = figsize, dpi = dpi)
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
        height = 80, 
        width = (endSyncTimes - startSyncTimes), 
        color = 'tab:green', 
        alpha = 0.2,
        label = 'Sync state'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Framerate (fps)')
    plt.legend()
    plt.tight_layout()
    plt.show()