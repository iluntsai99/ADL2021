#!bin/bash

if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi
mkdir -p ./ckpt/intent
mkdir -p ./ckpt/slot
wget https://www.dropbox.com/s/vpdc971afyrdfbs/91377.pt?dl=1 -O ./ckpt/intent/91377.pt
wget https://www.dropbox.com/s/idgn6uv2fjmogpu/78713.pt?dl=1 -O ./ckpt/slot/78713.pt