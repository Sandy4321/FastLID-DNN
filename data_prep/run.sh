#!/bin/bash

. ./cmd.sh
. utils/parse_options.sh

if [ $1 == "prep" ]; then
    echo "Starting from data prep"
    stage=1
elif [ $1 == "mfcc" ]; then
    echo "Starting from MFCC feature generation"
    stage=2
elif [ $1 == "vad" ]; then
    echo "Starting from voice-activity detection (VAD)"
    stage=3
elif [ $1 == "delta" ]; then
    echo "Starting from delta and delta-delta feature generation"
    stage=4
elif [ $1 == "label" ]; then
    echo "Starting from feature labeling"
    stage=5
else
    echo "Must specify a starting point: must be one of prep, mfcc, delta or label"
    exit 1
fi

if [ $2 == "3" ]; then
    echo "3-second utterances"
    utt_duration=3
elif [ $2 == "10" ]; then
    echo "10-second utterances"
    utt_duration=10
elif [ $2 == "30" ]; then
    echo "30-second utterances"
    utt_duration=30
else
    echo "Must specify an utterance duration: must be one of 3, 10 or 30"
    exit 1
fi

if [ $# -ne 3 ]; then
    # Process our features from the LRE 2003 dataset
    data=`pwd`/data
    data_train=$data/train_${utt_duration}
    data_test=$data/test_${utt_duration}
    featdir=`pwd`/feats_${utt_duration}_nodither

    if [ $stage -le 1 ]; then
        echo "Preparing data..."
        # convert sphere data to wav files, along with proper scp and ark files
        local/lre03_data_prep.sh $utt_duration || exit 1;
        echo "Data prepared!"
    fi

    if [ $stage -le 2 ]; then
        echo "Making MFCCs..."
        # Now make MFCC features.
        steps/make_mfcc.sh --cmd "$train_cmd" --nj $NUM_JOBS --mfcc-config conf/mfcc.conf \
            $data_train exp/make_feats/train $featdir || exit 1;
        steps/make_mfcc.sh --cmd "$train_cmd" --nj $NUM_JOBS --mfcc-config conf/mfcc.conf \
            $data_test exp/make_feats/test $featdir || exit 1;
        echo "MFCCs complete!"
    fi

    if [ $stage -le 3 ]; then
        # Tests showed that >98.5% of speech in LRE 2003 is tagged as activity -
        # little gain to be had for doing VAD here...
        echo "Voice-activity detection currently disabled"
        # echo "Running voice-activity detection..."
        # voice-activity detection to remove silence from utterances
        # local/vad.sh || exit 1;
        # echo "Voice-activity detection complete!"
    fi

    if [ $stage -le 4 ]; then
        echo "Making delta and delta-delta MFCCs..."
        add-deltas --delta-order=2 scp:$data_train/feats.scp ark,t:$featdir/features_train.ark
        add-deltas --delta-order=2 scp:$data_test/feats.scp ark,t:$featdir/features_test.ark
        echo "Delta and delta-delta MFCCs complete!"
    fi

    if [ $stage -le 5 ]; then
        echo "Making labeled feature set..."
        python local/label_features.py $featdir/features_train.ark $data_train/utt2lang
        python local/label_features.py $featdir/features_test.ark $data_test/utt2lang
        echo "Labeled feature set complete!"
    fi
else
    # Process a live-recorded wav file
    wav_file=$3

    data=`pwd`/data
    data_wav=$data/wav_data
    featdir=`pwd`/wav_feats

    if [ $stage -le 1 ]; then
        echo "Preparing data..."
        # process our proper scp and ark files
        local/wav_data_prep.sh $wav_file || exit 1;
        echo "Data prepared!"
    fi

    if [ $stage -le 2 ]; then
        echo "Making MFCCs..."
        # Now make MFCC features.
        steps/make_mfcc.sh --cmd "$demo_cmd" --nj 1 --mfcc-config conf/mfcc.conf \
            $data_wav exp/make_feats/$wav_file $featdir || exit 1;
        echo "MFCCs complete!"
    fi

    if [ $stage -le 3 ]; then
        # Tests showed that >98.5% of speech in LRE 2003 is tagged as activity -
        # little gain to be had for doing VAD here...
        echo "Voice-activity detection currently disabled"
        # echo "Running voice-activity detection..."
        # voice-activity detection to remove silence from utterances
        # local/vad.sh || exit 1;
        # echo "Voice-activity detection complete!"
    fi

    if [ $stage -le 4 ]; then
        echo "Making delta and delta-delta MFCCs..."
        add-deltas --delta-order=2 scp:$data_wav/feats.scp ark,t:$featdir/features.ark
        echo "Delta and delta-delta MFCCs complete!"
    fi

    if [ $stage -le 5 ]; then
        echo "Formatting feature set..."
        python local/format_features.py $featdir/features.ark
        echo "Formatted feature set complete!"
    fi
fi

echo "All done!"
exit 0;
