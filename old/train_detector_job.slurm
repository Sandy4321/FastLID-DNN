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
#SBATCH -J train_detector
. /etc/profile.d/modules.sh  
module load cuda/8.0
module load engaging/OpenBLAS/0.2.14
module load engaging/lua/5.3.2
module load engaging/torch/20160128

echo "TRAINING DNN MODEL FOR IN/OUT OF SET DETECTION"
optim=Adam
epochs=500
batchsize=32
filename=/pool001/atitus/FastLID-DNN/models/detector_1k_1k_256_${optim}_e${epochs}_b${batchsize}
threads=16
th train_detector.lua --netFilename="${filename}" --gpu -o $optim -e $epochs -t $threads -b $batchsize
echo "COMPLETED TRAINING"
