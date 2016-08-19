#!/bin/bash
#set -e
#python2.7 scripts/download.py

glove_dir="data/embedding"
glove_pre="glove.twitter.27B"
glove_dim="200d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    th scripts/convert-wordvecs.lua $glove_dir/$glove_pre.$glove_dim.txt \
        $glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi
