#!/bin/bash

# separate complete base in 60% of training and 40% of test
# process is done 10 times to avoid ...
for (( i=0; i < 10; i++ ))
do
  sort --random-sort complete_base  > temp
  head -5460 temp > train$i
  tail -3641 temp > test$i
  cat train$i | grep -o '.$' > train${i}-labels
  cat test$i | grep -o '.$' > test${i}-labels

  cat train$i | cut -d" " -f1-100 > temp
  mv temp train${i}
  cat test$i | cut -d" " -f1-100 > temp
  mv temp test${i}
done

#mkdir custom_bases
mv train* test* custom_bases
