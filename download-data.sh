#!/bin/bash


git clone https://github.com/ncg-task/training-data
git clone https://github.com/ncg-task/trial-data
#git clone https://github.com/ncg-task/evaluation-phase1

rm training-data/README.md
rm trial-data/README.md
#rm evaluation-phase1/README.md
mv training-data training-data-master
#mv evaluation-phase1 evaluation-phase1-master

cp -r trial-data/* training-data-master
