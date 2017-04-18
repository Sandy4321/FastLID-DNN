#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

dir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils

cd $dir

# Create scp's with wav's
awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp

# All done!
echo "VAD-based pruning succeeded"
