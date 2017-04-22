#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

data=`pwd`/data
train=$data/train
test=$data/test
local=`pwd`/local

# Run raw VAD to determine frames with activity
compute-vad scp:$train/feats.scp ark,t:$train/train_vad.ark
compute-vad scp:$test/feats.scp ark,t:$test/test_vad.ark

# TODO: use other script to get just the features in frames with activity

# All done!
echo "VAD-based pruning succeeded"
