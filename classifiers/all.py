#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn import model_selection
import sys


class classifiers():

    def __init__(self):
        dt = open(sys.argv[1])
        dt_labels = open(sys.argv[2])

        data = np.loadtxt(dt)
        data_labels = np.loadtxt(dt_labels)

        classifiers = [
            DecisionTreeClassifier(max_features=24, max_depth=10),
            KNeighborsClassifier(n_neighbors=5),
            #GaussianNB(),
            RandomForestClassifier(n_estimators=59, max_features=5),
            SVC(gamma=2, C=46, class_weight='balanced'),
            SVC(gamma=2, C=46, kernel='linear', class_weight='balanced', decision_function_shape='ovr', probability=True)
          ]

        names = [
            'Decision tree',
            'KNN',
            #'Naive Bayes',
            'Random forests',
            'SVM',
            'Linear kernel SVM'
        ]

        scores = []

        for clf, name in zip(classifiers, names):
            #print(clf)
            print(name)
            cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
            scs = model_selection.cross_val_score(clf, data, data_labels, cv=cv)
            print(scs)
            print("Acurácia média: %0.2f (+/- %0.2f)" % (scs.mean(), scs.std()*2 ) )
            print("Desvio padrão: %f" %(scs.std()))
            #scores.append(list(scs))

        clf = VotingClassifier(estimators=[('rf', classifiers[2]), ('svm', classifiers[4])], voting='hard')
        print(clf)
        cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
        scs = model_selection.cross_val_score(clf, data, data_labels, cv=cv)
        print(scs)
        print("Acurácia média: %0.2f (+/- %0.2f)" % (scs.mean(), scs.std()*2 ) )


if __name__ == "__main__":
    classifiers()
