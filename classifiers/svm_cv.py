#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import LinearSVC
from sklearn import model_selection, metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import sys

class svm():

    def __init__(self):
        dt = open(sys.argv[1])
        dt_labels = open(sys.argv[2])

        data = np.loadtxt(dt)
        data_labels = np.loadtxt(dt_labels)

        clf = LinearSVC(multi_class='ovr', class_weight='balanced', dual=False)
        clf.fit(data, data_labels)

        cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
        scores = model_selection.cross_val_score(clf, data, data_labels, cv=cv)

        #print('scores ', scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        res = model_selection.cross_val_predict(clf, data, data_labels, cv=10)
        print('acc pred ', metrics.precision_recall_fscore_support(data_labels, res))

        #print(classification_report(data_labels, res))

if __name__ == "__main__":
    svm()
