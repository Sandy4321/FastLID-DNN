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

# NIST Language Recognition Evaluation (LRE) 2003 datasets
lre03=/pool001/atitus/FastLID-DNN/data_prep/data/2003_nist_lrd
lre03_train=$lre03/data/lid96e1/test/
lre03_test=$lre03/data/lid96d1/test/
#lre03_train_n=1503
lre03_train_n=3005
lre03_test_n=1174

# Set up training and test data using only 3-second utterances
#find $lre03_train/3 -name '*.sph' > $dir/train.flist
#n=`cat $dir/train.flist | wc -l`
#[ $n -eq $lre03_train_n ] || echo "Unexpected number of training files $n versus $lre03_train_n"
#cp $lre03_train/3/seg_lang.ndx $dir/train_lang.ndx

#find $lre03_test/3 -name '*.sph' > $dir/test.flist
#n=`cat $dir/test.flist | wc -l`
#[ $n -eq $lre03_test_n ] || echo "Unexpected number of test files $n versus $lre03_test_n"
#cp $lre03_test/3/seg_lang.ndx $dir/test_lang.ndx

# Set up training for 3- and 10-second utterances in training set
find $lre03_train/3 -name '*.sph' > $dir/train3.flist
find $lre03_train/10 -name '*.sph' > $dir/train10.flist
#find $lre03_train/30 -name '*.sph' > $dir/train30.flist
#cat $dir/train3.flist $dir/train10.flist $dir/train30.flist > $dir/train.flist
cat $dir/train3.flist $dir/train10.flist > $dir/train.flist
n=`cat $dir/train.flist | wc -l`
[ $n -eq $lre03_train_n ] || echo "Unexpected number of training files $n versus $lre03_train_n"
cp $lre03_train/3/seg_lang.ndx $dir/train3_lang.ndx
cp $lre03_train/10/seg_lang.ndx $dir/train10_lang.ndx
#cp $lre03_train/30/seg_lang.ndx $dir/train30_lang.ndx
#cat $dir/train3_lang.ndx $dir/train10_lang.ndx $dir/train30_lang.ndx > $dir/train_lang.ndx
cat $dir/train3_lang.ndx $dir/train10_lang.ndx > $dir/train_lang.ndx

# Mark the languages we care about --- all others are out-of-set
languages=( english german mandarin )

#for x in train test; do
#    # get scp file that has utterance-ids and maps to the sphere file.
#    cat $dir/$x.flist | perl -ane 'm|/(..)/([0-9a-z]{4})\.sph| || die "bad line $_"; print "$2 $_"; ' \
#     | sort > $dir/${x}_sph.scp
#    # turn it into one that has a valid .wav format in the modern sense (i.e. RIFF format, not sphere).
#    # This file goes into its final location
#    mkdir -p $data/$x
#    awk '{printf("%s '$sph2pipe' -p -f wav %s |\n", $1, $2);}' < $dir/${x}_sph.scp > data/$x/wav.scp
#
#    # now get the "utt2lang" file that says, for each utterance, the language or out-of-set
#    printf "%s\n" "${languages[@]}" > $data/$x/languages
#    python $local/utt2lang.py $data/$x $dir/${x}_lang.ndx
#    sort $data/$x/utt2lang_unsorted > $data/$x/utt2lang
#
#    # create the file that maps from language to utterance-list.
#    python $local/utt2lang_to_lang2utt.py $data/$x
#
#    # Now to create files required for built-in Kaldi stuff
#    # get the "utt2spk" file that says, for each utterance, the speaker name.
#    # because LRE 2003 does not have speaker names, we simply map utterance ids to speakers here.
#    python $local/utt2spk_dumb.py $data/$x
#    # create the file that maps from "speaker" to utterance-list.
#    utils/utt2spk_to_spk2utt.pl <$data/$x/utt2spk >$data/$x/spk2utt
#done

# get scp file that has utterance-ids and maps to the sphere file.
cat $dir/train.flist | perl -ane 'm|/(..)/([0-9a-z]{4})\.sph| || die "bad line $_"; print "$2 $_"; ' \
 | sort > $dir/train_sph.scp
# turn it into one that has a valid .wav format in the modern sense (i.e. RIFF format, not sphere).
# This file goes into its final location
mkdir -p $data/train
awk '{printf("%s '$sph2pipe' -p -f wav %s |\n", $1, $2);}' < $dir/train_sph.scp > data/train/wav.scp

# now get the "utt2lang" file that says, for each utterance, the language or out-of-set
printf "%s\n" "${languages[@]}" > $data/train/languages
python $local/utt2lang.py $data/train $dir/train_lang.ndx
sort $data/train/utt2lang_unsorted > $data/train/utt2lang

# create the file that maps from language to utterance-list.
python $local/utt2lang_to_lang2utt.py $data/train

# Now to create files required for built-in Kaldi stuff
# get the "utt2spk" file that says, for each utterance, the speaker name.
# because LRE 2003 does not have speaker names, we simply map utterance ids to speakers here.
python $local/utt2spk_dumb.py $data/train
# create the file that maps from "speaker" to utterance-list.
utils/utt2spk_to_spk2utt.pl <$data/train/utt2spk >$data/train/spk2utt

# All done!
echo "Data preparation succeeded"
