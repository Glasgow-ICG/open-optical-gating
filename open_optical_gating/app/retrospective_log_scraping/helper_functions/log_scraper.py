import numpy as np
import json

def LogFileScraper(logFilePath, keyDictPath, zeroTimes = True):
    """
    A function to iteratively search each line of a given log file for 2 key phrases contained in 
    the associated 'logger_keys.json' file. 
    
    If lines contain either of these strings, the split string indices included in the same json
    are used to extract the relevant time stamp, phase, target phase, and trigger time values, which 
    are then added to a dictionary.
    
    Inputs:
        logFilePath = path to the desired log file
        keyDictPath = path to the relevant key json file
        zeroTimes = boolean to decide whether or not to recentre the timestamps to time t = 0
    
    Outputs:
        scrapedDict = dictionary containing keys {timeStamps, phases, targetPhases, triggerTimes}
    """

    keyDict = json.load(open(keyDictPath))
    scrapedDictKeys = [
        'timeStamps',
        'states',
        'phases',
        'targetPhases',
        'predictedTimes',
        'triggerTimes'
    ]
    scrapedDict= dict(zip(scrapedDictKeys, [[] for i in range(len(scrapedDictKeys))]))
    with open(logFilePath) as log:
        for line in log.readlines():
            if keyDict["keyTypeA"] in line:
                scrapedDict["timeStamps"].append(float(line.split()[keyDict["timeStampIndex"]]))
                scrapedDict["states"].append(line.split()[keyDict["stateIndex"]])
                
                if not line.split()[keyDict["phaseIndex"]] == 'None':
                    scrapedDict["phases"].append(float(line.split()[keyDict["phaseIndex"]]))
                    
                if not line.split()[keyDict["predictedTime"]] == 'None':
                    scrapedDict["predictedTimes"].append(float(line.split()[keyDict["predictedTime"]]))
                else:
                    scrapedDict["predictedTimes"].append(-1)

                if not line.split()[keyDict["targetPhaseIndex"]] == 'None':
                    scrapedDict["targetPhases"].append(float(line.split()[keyDict["targetPhaseIndex"]]))
           
            if keyDict["keyTypeB"] in line:
                scrapedDict["triggerTimes"].append(float(line.split()[keyDict["triggerTimeSentIndex"]]))

    scrapedDict["phases"] = np.asarray(scrapedDict["phases"])
    scrapedDict["targetPhases"] = np.asarray(scrapedDict["targetPhases"])
    scrapedDict["states"] = np.asarray(scrapedDict["states"])
    
    if zeroTimes:
        scrapedDict["triggerTimes"] = np.asarray(scrapedDict["triggerTimes"]) - np.asarray(scrapedDict["timeStamps"][0])
        scrapedDict["predictedTimes"] = np.asarray(scrapedDict["predictedTimes"]) - np.asarray(scrapedDict["timeStamps"][0])
        # Zero the timestamps last, as we were relying on its absolute value until this point
        scrapedDict["timeStamps"] = np.asarray(scrapedDict["timeStamps"]) - np.asarray(scrapedDict["timeStamps"][0])
    else:
        scrapedDict["triggerTimes"] = np.asarray(scrapedDict["triggerTimes"])
        scrapedDict["predictedTimes"] = np.asarray(scrapedDict["predictedTimes"])
        scrapedDict["timeStamps"] = np.asarray(scrapedDict["timeStamps"])

    return scrapedDict
