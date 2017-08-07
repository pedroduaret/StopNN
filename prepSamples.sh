#!/bin/bash

FILES="Wjets_200to400 Wjets_400to600 Wjets_600to800 T2DegStop_300_270"

for file in $FILES; do
  echo "Checking files for sample ${file}"
  if [[ ! -f ${file}_test.root ]]; then
    echo "  The test file does not exist, downloading it"
    scp cbeiraod@fermi.ncg.ingrid.pt:~/local-area/Stop4Body/nTuples_v2017-06-05_SysVar_test/${file}.root ./${file}_test.root
  fi
  if [[ ! -f ${file}_train.root ]]; then
    echo "  The train file does not exist, downloading it"
    scp cbeiraod@fermi.ncg.ingrid.pt:~/local-area/Stop4Body/nTuples_v2017-06-05_SysVar_train/${file}.root ./${file}_train.root
  fi
done

setupRoot
for file in $FILES; do
  echo "Checking files of sample ${file} for skimming"
  if [[ ! -f ${file}_test_skimmed.root ]]; then
    echo "  The test sample has not been skimmed, skimming now"
    root -b -q -l skimmer.cc+\(\"${file}_test\"\)
  fi
  if [[ ! -f ${file}_train_skimmed.root ]]; then
    echo "  The train sample has not been skimmed, skimming now"
    root -b -q -l skimmer.cc+\(\"${file}_train\"\)
  fi
done




