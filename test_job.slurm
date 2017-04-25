#!/bin/bash  
#SBATCH --gres=gpu:1  
#SBATCH --mem=64000
#SBATCH -N 1  
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --reservation=jimglass_class_testing
#SBATCH -p sched_engaging_default
#SBATCH --time=01:00:00  
#SBATCH --exclusive  
#SBATCH -J test
. /etc/profile.d/modules.sh  
module load cuda/8.0
module load engaging/OpenBLAS/0.2.14
module load engaging/lua/5.3.2
module load engaging/torch/20160128

echo "TESTING DNN MODEL"
optim=Adam
epochs=100
batchsize=128
network=/pool001/atitus/FastLID-DNN/models/1k_1k_${optim}_e${epochs}_b${batchsize}
threads=16
echo "TESTING NETWORK WITH FILENAME ${network}"
th test.lua --network="${network}" --gpu --full -t $threads
echo "COMPLETED TESTING"
