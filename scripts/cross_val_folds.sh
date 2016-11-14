#!/bin/bash

set -x

# separate complete base in 60% of training and 40% of test
# process is done 10 times to avoid ...
for (( i=0; i < 10; i++ ))
do
  sort --random-sort ../complete_base  > temp
  head -444 temp > train$i
  tail -296 temp > test$i

  total=$(head -1 train$i | wc -w)
  ((n_features = total - 1))
  echo $n_features

  cat train$i | cut -d' ' -f$total > train${i}-labels
  cat test$i | cut -d' ' -f$total > test${i}-labels

  cat train$i | cut -d" " -f1-$n_features > temp
  mv temp train${i}
  cat test$i | cut -d" " -f1-$n_features > temp
  mv temp test${i}
done

#mkdir custom_bases
mv train* test* ../custom_bases
