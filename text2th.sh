#!/bin/bash
glove_dir="data/embedding"
glove_pre="glove.6B"
glove_dim="200d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    th scripts/text2th.lua $glove_dir/$glove_pre.$glove_dim.txt \
        $glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi
