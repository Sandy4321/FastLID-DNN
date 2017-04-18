#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.


if [ $# -ne 1 ]; then
   echo "Argument should be the sphere data directory, see ../run.sh for example."
   exit 1;
fi
sph_data=$1


dir=`pwd`/data/local/data
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

cd $dir

# Create scp's with wav's
python $local/ndx2flist.py $lre07/docs/seg.ndx
awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp

# All done!
echo "Data preparation succeeded"
