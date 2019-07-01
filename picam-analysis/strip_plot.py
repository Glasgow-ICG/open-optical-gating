import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skimage
import seaborn as sns
import pickle

# script to define necessary functions to find and plot the SSIM metric of sets of images 

def reformat(ssim,imnum, label = 'format',):

    # takes the SSIM of a set of images as a list, as well as the number of images and creates a dictionary
    # that is compatible with a seaborn strip plot

        plot_dict = {}
        plot_dict[label] = []
        plot_dict['SSIM'] = []

        for i in ssim.keys():
            plot_dict[label].extend([i]*imnum)
            plot_dict['SSIM'].extend(ssim[i])
        
        return plot_dict

def make_plot(filename):

    # makes the stripplot from a pickled file of images as numpy arrays
    # pickle may not be the best way to store these arrays, but it was the most convenient to use at the time of writing
    
    file = open(filename, 'rb')
    images = pickle.load(file) 

    imnum = len(images['rgb']) ##CHANGE THIS TO GENERAL CASE 
    print(imnum)
    ssim = {}

    for i in range(imnum):
        for j in images.keys():      
        
            if i == 0:
                ssim[j] = [0]*imnum            
            ssim[j][i] = skimage.measure.compare_ssim(images[j]["image0"], images[j]["image%d"%(i)], multichannel = True)      
    
    s = reformat(ssim,imnum)
    print(type(s))

    sns.set(style = "whitegrid")
    ax = sns.stripplot(x = "format",y = "SSIM", data = s)

    plt.show()
    return plt
