from image_loading import *
import scipy.misc
from matplotlib import pyplot as plt

def AverageImages(srcPath, combinedPath):
    # Calculate a 4-point average over Alex's compressive data to filter out 50Hz interference

    # WARNING: on both python 2 and 3, scipy seems to be saving as 8-bit tiffs despite my best efforts, which is not what I want!
    # That's weird though, because I thought I had got this to work previously. Not sure what to do now...
    #
    # -> Mar 2018: I can update to new code using tifffile, which should be able to save 16-bit tiffs (see image_saving.py)
    assert(0)
    
    (images, dummy) = LoadAllImages(srcPath)
    for i in range(len(images)-3):
        images[i].image = (images[i].image + images[i+1].image + images[i+2].image + images[i+3].image) / 4.0
        images[i].image = images[i].image.astype('uint16')
        scipy.misc.imsave(images[i]._path, images[i].image)
        if (i == 0):
            combinedImage = images[i].image.copy()
        else:
            combinedImage = np.append(combinedImage, images[i].image, axis=1)
    scipy.misc.imsave(combinedPath, combinedImage)
    # Note that this will put the plists out of step, which may or may not be a problem depending on the dataset
    # and whether or not it is sync'd to a brightfield channel.

if __name__ == "__main__":
    #AverageImages('/Users/jonny/Movies/Example_compr_sens_datasets/New datasets/Darkfield/df_fish3_maskA_photodiodetifs_2000to3499_averaged')
    #AverageImages('/Users/jonny/Movies/Example_compr_sens_datasets/New datasets/Brightfield/bf_fish3_maskA_photodiodetifs_2000to3499_averaged', '/Users/jonny/Movies/Example_compr_sens_datasets/New datasets/Brightfield/averaged.tif')
    AverageImages('/Users/jonny/Movies/Example_compr_sens_datasets/New datasets/Brightfield/bf_fish3_maskA_photodiodetifs_2000to3499', '/Users/jonny/Movies/Example_compr_sens_datasets/New datasets/Brightfield/averaged.tif')
