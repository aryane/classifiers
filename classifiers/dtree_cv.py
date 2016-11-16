#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import model_selection
import sys


class decision_tree():

    def __init__(self):
        dt = open(sys.argv[1])
        dt_labels = open(sys.argv[2])

        data = np.loadtxt(dt)
        data_labels = np.loadtxt(dt_labels)

        clf = DecisionTreeClassifier(max_features=24, max_depth=10)

        cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
        scores = model_selection.cross_val_score(clf, data, data_labels, cv=cv)

        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        res = model_selection.cross_val_predict(clf, data, data_labels, cv=10)
        print('acc pred ', accuracy_score(data_labels, res))
        #print(classification_report(data_labels, res))


if __name__ == "__main__":
    decision_tree()
