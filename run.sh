#!/bin/bash

# extract features from images
#python create_bases.py

# separate complete base in 60% of training and 40% of test
# process is done 10 times to avoid ...
./custom_bases/generate_train_test.sh
