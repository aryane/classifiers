#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

class naive_bayes():

    def __init__(self):
        training = open(sys.argv[1])
        training_labels = open(sys.argv[2])
        test = open(sys.argv[3])
        test_label = open(sys.argv[4])

        train = np.loadtxt(training) #training
        train_labels = np.loadtxt(training_labels) #labels
        tst = np.loadtxt(test)
        tst_label = np.loadtxt(test_label)

        clf = GaussianNB()
        clf = clf.fit(train, train_labels)

        #print(clf.predict(tst[20])[0], tst_label[20])
        res = clf.predict(tst)

        print(confusion_matrix(tst_label, res))
        print(accuracy_score(tst_label, res))

if __name__ == "__main__":
    naive_bayes()
