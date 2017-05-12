#!/bin/bash  
#SBATCH --gres=gpu:1  
#SBATCH --mem=64000
#SBATCH -N 1  
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --reservation=jimglass_class_testing
#SBATCH -p sched_engaging_default
#SBATCH --time=02:00:00  
#SBATCH --exclusive  
#SBATCH -J posteriors
. /etc/profile.d/modules.sh  
module load cuda/8.0
module load engaging/OpenBLAS/0.2.14
module load engaging/lua/5.3.2
module load engaging/torch/20160128

languages=german_mandarin
model_name=1k_1k_512_64_1k_${languages}
network=/pool001/atitus/FastLID-DNN/models/${model_name}
threads=16
echo "EVALUATING POSTERIORS FOR NETWORK WITH MODEL NAME ${model_name}"
th posteriors.lua --network="${network}" --gpu -t $threads --languages="${languages}"
echo "DONE EVALUATING POSTERIORS"
