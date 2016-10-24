#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import sys

class knn():

    def __init__(self):
        training = open(sys.argv[1])
        training_labels = open(sys.argv[2])
        test = open(sys.argv[3])
        print(sys.argv[4])
        test_label = open(sys.argv[4])
        k = int(sys.argv[5])

        train = np.loadtxt(training) #training
        train_labels = np.loadtxt(training_labels) #labels
        tst = np.loadtxt(test)
        tst_label = np.loadtxt(test_label)

        clf = KNeighborsClassifier(n_neighbors=k)
        clf = clf.fit(train, train_labels)

        #print(clf.predict(tst[20])[0], tst_label[20])
        res = clf.predict(tst)

        print(classification_report(tst_label, res))
        print(confusion_matrix(tst_label, res))
        print(accuracy_score(tst_label, res))

if __name__ == "__main__":
    knn()
