#!/bin/bash
set -eu -o pipefail

### GloVe vectors ###
if [ ! -d glove ]
then
  mkdir glove
  cd glove
  wget http://nlp.stanford.edu/data/glove.6B.zip
  unzip glove.6B.zip
  cd ..
fi

### SQuAD ###
mkdir -p data
cd data
if [ ! -d squad ]
then
  mkdir squad
  cd squad
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  cd ..
fi
cd ..

### nectar
if [ ! -d "nectar" ]
then
  git clone https://github.com/robinjia/nectar.git
  cd nectar 
  ./pull_dependencies.sh
  cd ..
  ln -s ../../nectar/nectar src/py
fi
