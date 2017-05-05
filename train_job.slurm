#!/bin/bash  
#SBATCH --gres=gpu:1  
#SBATCH --mem=64000
#SBATCH -N 1  
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --reservation=jimglass_class_testing
#SBATCH -p sched_engaging_default
#SBATCH --time=12:00:00  
#SBATCH --exclusive  
#SBATCH -J train
. /etc/profile.d/modules.sh  
module load cuda/8.0
module load engaging/OpenBLAS/0.2.14
module load engaging/lua/5.3.2
module load engaging/torch/20160128

echo "TRAINING DNN MODEL"
optim=Adam
epochs=100
batchsize=128
weight_decay=0.00005
network=/pool001/atitus/FastLID-DNN/models/1k_1k_64_1k_es_wd${weight_decay}
filename=/pool001/atitus/FastLID-DNN/models/1k_1k_64_1k_es_wd${weight_decay}
threads=16
echo "TRAINING NETWORK WITH FILENAME ${filename}"
th train.lua --netFilename="${filename}" --gpu -o $optim -e $epochs -t $threads -b $batchsize --earlyStopping -w ${weight_decay}
#th train.lua -n $network --netFilename="${filename}" --gpu -o $optim -e $epochs -t $threads -b $batchsize --earlyStopping -w ${weight_decay}
echo "COMPLETED TRAINING"
