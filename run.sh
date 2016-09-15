#!/bin/bash

# extract features from images
#python create_bases.py

# separate complete base in 60% of training and 40% of test
# process is done 10 times to avoid ...
#./custom_bases/generate_train_test.sh

#
for (( i=0; i < 10; i++ )); do
  python classifiers/knn.py custom_bases/train${i} custom_bases/train${i}-labels custom_bases/test${i} custom_bases/test${i}-labels 5 > results/${i}knn
  python classifiers/dtree.py custom_bases/train${i} custom_bases/train${i}-labels custom_bases/test${i} custom_bases/test${i}-labels > results/${i}dtree
  python classifiers/naivebayes.py custom_bases/train${i} custom_bases/train${i}-labels custom_bases/test${i} custom_bases/test${i}-labels > results/${i}naivebayes
done
