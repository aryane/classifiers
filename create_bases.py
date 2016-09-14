#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage.color import rgb2gray

import os

class classifiers_comp():

    def __init__(self):
        self.generate_complete_base('complete_base')

    def histogram_vector(self, file_prefix, img):
        try:
            img = rgb2gray(io.imread(file_prefix+img))
        except:
            return ""

        hist = np.histogram(img, 100)
        desc = []
        for i in hist[0]:
            desc.append(i)
        return(" ".join(str(x) for x in desc))


    def generate_complete_base(self, file_name):
        base_file = open(file_name, 'w')

        dirs = ['CI171TrainVal/01/', 'CI171TrainVal/02/', 'CI171TrainVal/03/', 'CI171TrainVal/04/', 'CI171TrainVal/05/', 'CI171TrainVal/06/', 'CI171TrainVal/07/', 'CI171TrainVal/08/', 'CI171TrainVal/09/']


        ''' melhoria: será que dá pra usar só a lista dirs acima? '''
        dir1 = os.listdir('CI171TrainVal/01')
        dir2 = os.listdir('CI171TrainVal/02')
        dir3 = os.listdir('CI171TrainVal/03')
        dir4 = os.listdir('CI171TrainVal/04')
        dir5 = os.listdir('CI171TrainVal/05')
        dir6 = os.listdir('CI171TrainVal/06')
        dir7 = os.listdir('CI171TrainVal/07')
        dir8 = os.listdir('CI171TrainVal/08')
        dir9 = os.listdir('CI171TrainVal/09')

        ''' Create training base with intercalary classes/labels '''
        for a, b, c, d, e, f, g, h, i in map(None, dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8, dir9):
            for pref, img_name in zip(dirs, [a, b, c, d, e, f, g, h, i]):
                if img_name != None:
                    vec = self.histogram_vector(pref, img_name)
                    if vec:
                        vec += " " + (pref[-2])
                        base_file.write(vec+'\n')


if __name__ == "__main__":
    classifiers_comp()
