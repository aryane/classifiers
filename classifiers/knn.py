#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import sys
import math

class knn():

    def __init__(self):
        #training, test, vec_size = self.parse_args()

        training = open(sys.argv[1])
        training_labels = open(sys.argv[2])
        test = open(sys.argv[3])
        test_label = open(sys.argv[4])
        k = int(sys.argv[5])

        train = np.loadtxt(training) #training
        train_labels = np.loadtxt(training_labels) #labels
        tst = np.loadtxt(test)
        tst_label = np.loadtxt(test_label)

        clf = KNeighborsClassifier(n_neighbors=k)
        clf = clf.fit(train, train_labels)

        res = []
        correct = fail = 0
        for t, label in zip(tst, tst_label):
            pred = clf.predict(t)[0]
            res.append(pred)
            if clf.predict(t)[0] == label:
                correct += 1
            else:
                fail += 1

        print(confusion_matrix(tst_label, res))
        print(accuracy_score(tst_label, res))

if __name__ == "__main__":
    knn()
