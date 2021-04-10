#!bin/bash

mkdir -p ./ckpt/intent
mkdir -p ./ckpt/slot
wget https://www.dropbox.com/s/vpdc971afyrdfbs/91377.pt?dl=1 -O ./ckpt/intent/91377.pt
wget https://www.dropbox.com/s/cd48bmfq4sxbsc4/78552.pt?dl=1 -O ./ckpt/slot/78552.pt