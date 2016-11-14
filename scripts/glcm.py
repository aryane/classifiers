#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage.color import rgb2gray

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage.color import label2rgb

import os

class texture_representation():

    def __init__(self):
        self.generate_complete_base('complete_base')

    def generate_complete_base(self, file_name):
        images_dir = '../CI171TrainVal'

        ''' LBP settings '''
        radius = 2
        n_points = 8 * radius
        method = 'uniform'

        #base_file = open(file_name, 'w')

        dirs = os.listdir(images_dir) # list returned
        dirs.sort()

        ''' key: label, content: content of label dir '''
        d = {}
        for x in dirs:
            d[x] = os.listdir(images_dir+'/'+x)
            #d[x.replace(' ', '_')] = os.listdir(images_dir+'/'+x)

        ''' Create training base with intercalary classes/labels '''
        for label, file_names in d.items():
            for img_name in file_names:
                if img_name != None:
                    pref = images_dir+'/'+label+'/'
                    try:
                        img = skimage.io.imread(pref+img_name, as_grey=True)
                        img = skimage.img_as_ubyte(img)
                    except:
                        continue


                    glcm = greycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
                    corr = greycoprops(glcm, 'correlation')[0][0]
                    dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
                    contrast = greycoprops(glcm, 'dissimilarity')[0][0]
                    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
                    energy = greycoprops(glcm, 'energy')[0][0]
                    ASM = greycoprops(glcm, 'ASM')[0][0]
                    print("%s %s %s %s %s %s" %(str(corr), str(dissimilarity), str(contrast), str(homogeneity), str(energy), str(ASM)))
                    print( [corr,dissimilarity, contrast, homogeneity, energy, ASM])


                    contrast = dissimilarity = homogeneity = energy = correlation = ASM = []



                    print('*****************')

                    #n_bins = lbp.max() + 1
                    #hist, _ = np.histogram(lbp, normed=True, bins=n_bins,
                    #        range=(0, n_bins))

                    #desc = " ".join(str(x) for x in list(hist))
                    #desc += " " + label.partition(' ')[0] # add label number to end of descriptors
                    #base_file.write(desc+'\n')
                break;
            break;


if __name__ == "__main__":
    texture_representation()
