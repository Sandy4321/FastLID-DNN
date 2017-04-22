#!/bin/bash

. ./cmd.sh
. utils/parse_options.sh


# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

# convert sphere data to wav files, along with proper scp and ark files
local/lre03_data_prep.sh || exit 1;

# voice-activity detection to remove silence from utterances
# local/vad.sh || exit 1;

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

#for x in test_eval92 test_eval93 test_dev93 train_si284; do
#    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/$x || exit 1;
#    steps/compute_cmvn_stats.sh data/$x || exit 1;
#done

# All done!
exit 0;
