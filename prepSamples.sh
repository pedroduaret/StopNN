#!/bin/bash

FILES="Wjets_200to400 T2DegStop_300_270"

#for file in $FILES; do
#  echo "Downloading files of sample $file"
#  CMD="scp cbeiraod@fermi.ncg.ingrid.pt:~/local-area/Stop4Body/nTuples_v2017-06-05_SysVar_test/"$file".root ./"$file"_test.root"
#  eval $CMD
#  CMD="scp cbeiraod@fermi.ncg.ingrid.pt:~/local-area/Stop4Body/nTuples_v2017-06-05_SysVar_train/"$file".root ./"$file"_train.root"
#  eval $CMD
#done

setupRoot
for file in $FILES; do
  echo "Processing file: "$file"_test.root"
  CMD="root -b -q -l skimmer.cc+\(\\\""$file"_test\\\"\)"
  eval $CMD
  echo "Processing file: "$file"_train.root"
  CMD="root -b -q -l skimmer.cc+\(\\\""$file"_train\\\"\)"
  eval $CMD
done





#scp cbeiraod@fermi.ncg.ingrid.pt:~/local-area/Stop4Body/nTuples_v2017-06-05_SysVar_test/Wjets_200to400.root ./Wjets_200to400_test.root
