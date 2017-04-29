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
#SBATCH -J test_tandem
. /etc/profile.d/modules.sh  
module load cuda/8.0
module load engaging/OpenBLAS/0.2.14
module load engaging/lua/5.3.2
module load engaging/torch/20160128

echo "TESTING DNN TANDEM MODEL"
detector_network=/pool001/atitus/FastLID-DNN/models/detector_1k_1k_256_Adam_e500_b32
inset_network=/pool001/atitus/FastLID-DNN/models/no_oos_1k_512_256_Adam_e500_b32
threads=16
echo "TESTING TANDEM NETWORK WITH DETECTOR FILENAME ${detector_network} AND IN-SET FILENAME ${inset_network}"
th test_tandem.lua --detectorNet="${detector_network}" --insetNet="${inset_network}" --gpu -t $threads
echo "COMPLETED TESTING"
