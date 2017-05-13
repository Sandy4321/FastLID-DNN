#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

. ./path.sh # Needed for KALDI_ROOT

if [ $# -ne 1 ]; then
    echo "Need to specify wav file"
    exit 1;
fi

wav_file=$1
utt_id=1234     # Completely arbitrary

data=`pwd`/data
dir=$data/local/data
mkdir -p $dir

# get scp file that has utterance-ids and maps to the wav file.
# This file goes into its final location
mkdir -p $data/wav_data
echo "${utt_id} cat ${wav_file} |" > data/wav_data/wav.scp

# Now to create files required for built-in Kaldi stuff
# get the "utt2spk" file that says, for each utterance, the speaker name.
# We only have one utterance here - simply map our arbitrary utterance ID to itself here.
echo "$utt_id $utt_id" > $data/wav_data/utt2spk
# create the file that maps from "speaker" to utterance-list.
echo "$utt_id $utt_id" > $data/wav_data/spk2utt

# All done!
echo "Data preparation succeeded"
