#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

. ./path.sh # Needed for KALDI_ROOT

data=`pwd`/data
dir=$data/local/data
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

# Only do data prep for this utterance duration!
if [ $# -ne 1 ]; then
    echo "Need to specify utterance duration"
    exit 1
fi
utt_duration=$1

# NIST Language Recognition Evaluation (LRE) 2003 datasets
lre03=/pool001/atitus/FastLID-DNN/data_prep/data/2003_nist_lrd
lre03_train=$lre03/data/lid96e1/test/
lre03_test=$lre03/data/lid96d1/test/

lre03_train_n=0
lre03_test_n=0
if [ $utt_duration == "3" ]; then
    lre03_train_n=1503
    lre03_test_n=1174
elif [ $utt_duration == "10" ]; then
    lre03_train_n=1502
    lre03_test_n=1172
elif [ $utt_duration == "30" ]; then
    lre03_train_n=1492
    lre03_test_n=1147
else
    echo "Unexpected utterance duration $utt_duration"
    exit 1
fi
echo "$lre03_train_n training utterances"
echo "$lre03_test_n test utterances"

# Set up training and test data
find $lre03_train/${utt_duration} -name '*.sph' > $dir/train_${utt_duration}.flist
n=`cat $dir/train_${utt_duration}.flist | wc -l`
[ $n -eq $lre03_train_n ] || echo "Unexpected number of training files $n versus $lre03_train_n"
cp $lre03_train/${utt_duration}/seg_lang.ndx $dir/train_${utt_duration}_lang.ndx

find $lre03_test/${utt_duration} -name '*.sph' > $dir/test_${utt_duration}.flist
n=`cat $dir/test_${utt_duration}.flist | wc -l`
[ $n -eq $lre03_test_n ] || echo "Unexpected number of test files $n versus $lre03_test_n"
cp $lre03_test/${utt_duration}/seg_lang.ndx $dir/test_${utt_duration}_lang.ndx

# Mark the languages we care about --- all others are out-of-set
languages=( english german mandarin )


# Set up training and test files


for x in train test; do
    # get scp file that has utterance-ids and maps to the sphere file.
    cat $dir/${x}_${utt_duration}.flist | perl -ane 'm|/(..)/([0-9a-z]{4})\.sph| || die "bad line $_"; print "$2 $_"; ' \
     | sort > $dir/${x}_${utt_duration}_sph.scp
    # turn it into one that has a valid .wav format in the modern sense (i.e. RIFF format, not sphere).
    # This file goes into its final location
    mkdir -p $data/${x}_${utt_duration}
    awk '{printf("%s '$sph2pipe' -p -f wav %s |\n", $1, $2);}' < $dir/${x}_${utt_duration}_sph.scp > data/${x}_${utt_duration}/wav.scp

    # now get the "utt2lang" file that says, for each utterance, the language or out-of-set
    printf "%s\n" "${languages[@]}" > $data/${x}_${utt_duration}/languages
    python $local/utt2lang.py $data/${x}_${utt_duration} $dir/${x}_${utt_duration}_lang.ndx
    sort $data/${x}_${utt_duration}/utt2lang_unsorted > $data/${x}_${utt_duration}/utt2lang

    # create the file that maps from language to utterance-list.
    python $local/utt2lang_to_lang2utt.py $data/${x}_${utt_duration}

    # Now to create files required for built-in Kaldi stuff
    # get the "utt2spk" file that says, for each utterance, the speaker name.
    # because LRE 2003 does not have speaker names, we simply map utterance ids to speakers here.
    python $local/utt2spk_dumb.py $data/${x}_${utt_duration}
    # create the file that maps from "speaker" to utterance-list.
    utils/utt2spk_to_spk2utt.pl <$data/${x}_${utt_duration}/utt2spk >$data/${x}_${utt_duration}/spk2utt
done

# All done!
echo "Data preparation succeeded"
