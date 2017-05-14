#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Must specify wav file, network file, and number of languages"
    exit 1;
fi

wav_file=$1
net_file=$2
num_langs=$3

num_threads=4

# Process our wav file into the desired MFCCs, using the existing features script we have
demo_dir=`pwd`
data_prep_dir=$demo_dir/../data_prep/
cd $data_prep_dir
source path.sh
./run.sh prep 3 $wav_file
feat_file=$data_prep_dir/wav_feats/features.ark

# Run the demo code, now that we have our features!
cd $demo_dir
th demo.lua -n $net_file -t $num_threads --numLanguages=${num_langs} -f $feat_file