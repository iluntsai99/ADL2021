#!bin/bash
mkdir -p ./ckpt/ckpt_CS
mkdir -p ./ckpt/ckpt_QA
wget https://www.dropbox.com/s/ar9zvcgm1csk7a9/config.json?dl=1 -O ./ckpt/ckpt_CS/config.json
wget https://www.dropbox.com/s/9ljtdxkbth9w8ta/pytorch_model.bin?dl=1 -O ./ckpt/ckpt_CS/pytorch_model.bin
wget https://www.dropbox.com/s/y2pkgncgxhbrleg/config.json?dl=1 -O ./ckpt/ckpt_QA/config.json
wget https://www.dropbox.com/s/coqj4b1iuejlfpm/pytorch_model.bin?dl=1 -O ./ckpt/ckpt_QA/pytorch_model.bin