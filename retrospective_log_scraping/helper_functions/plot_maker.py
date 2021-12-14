import numpy as np
import matplotlib.pyplot as plt

def plot_triggers(timeStamps, phases, triggerTimes, targetPhases):
    """
    Plot the phase vs. time sawtooth line with trigger events.
    Note: The very first target phase is used to denote the target phase of the entire 
    sync. This is due to the fact that the actual target phase is decided with respect
    to the initial reference sequence.
    """
    plt.figure()
    plt.title("Zebrafish heart phase with trigger fires")
    plt.plot(
        timeStamps,
        phases % (2 * np.pi),
        label="Heart phase",
        color = "tab:blue"
    )
    plt.scatter(
        triggerTimes,
        np.full(max(len(triggerTimes), 0), targetPhases[0]),
        color= "tab:red",
        label= "Trigger fire",
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
    plt.figure(figsize = (6, 4))
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
    plt.figure(figsize = (6, 4))
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
    plt.figure(figsize = (6, 4))
    plt.title('Triggered phase error with time')
    plt.scatter(
        triggerTimes, 
        phaseErrors, 
        color = 'tab:green'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Phase error (rad)')
    plt.ylim(-np.pi, np.pi)
    plt.tight_layout()
    plt.show()