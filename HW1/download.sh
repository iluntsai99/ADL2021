#!bin/bash

if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi
mkdir -p ./ckpt/intent
mkdir -p ./ckpt/slot
wget https://www.dropbox.com/s/z5plt7psirkir2e/91822.pt?dl=0 -O ./ckpt/intent/intentModel.pt
wget https://www.dropbox.com/s/silpwjeaz1fspra/82252.pt?dl=1 -O ./ckpt/slot/slotModel.pt