# FastLID-DNN
Deep Neural Networks for Fast Automatic Language Identification over Small Language Sets

## To train network:
Change parameters as desired in `train.lua` and `train_job.slurm`, then submit using:
`sbatch train_job.slurm`

## To evaluate network:
Change parameters as desired in `evaluate.lua` and `evaluate_job.slurm`, then submit using:
`sbatch evaluate_job.slurm`



### NOTE: there is a known issue in Kaldi (https://github.com/kaldi-asr/kaldi/issues/717) where there is some non-determinism introduced to feature extraction via dithering. Use --dither=0 in data_prep/conf/mfcc.conf to get deterministic MFCC results.
