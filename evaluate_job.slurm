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
#SBATCH -J evaluate
. /etc/profile.d/modules.sh  
module load cuda/8.0
module load engaging/OpenBLAS/0.2.14
module load engaging/lua/5.3.2
module load engaging/torch/20160128

echo "EVALUATING DNN MODEL"
model_name=1k_1k_512_64_1k_es
#selected_utt=01yr       # English (apparently??)
confidence_threshold=0.99
network=/pool001/atitus/FastLID-DNN/models/${model_name}
threads=16
echo "EVALUATING NETWORK WITH MODEL NAME ${model_name}"
th evaluate.lua --network="${network}" --gpu -t $threads --confidenceThreshold="${confidence_threshold}"
#th evaluate.lua --network="${network}" --gpu -t $threads --selectedUtt="${selected_utt}"
echo "COMPLETED EVALUATION"
