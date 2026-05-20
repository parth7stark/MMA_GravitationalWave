#!/bin/bash
#SBATCH --mem=100g                              # required number of memory
#SBATCH --nodes=1                               # nodes required for whole simulation

#SBATCH --cpus-per-task=32                       # CPUs for each task
## SBATCH --gpus-per-task=1                      # Uncomment if using gpu
##SBATCH --ntasks-per-node=1                     # Uncomment if using gpu 

#SBATCH --partition=cpu                    # <- or if Delta one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8

##SBATCH --gpus-per-node=1      # Uncomment if using gpu         # number of GPUs you want to use on 1 node
##SBATCH --gpu-bind=none        # Uncomment if using gpu


#SBATCH --job-name=MMA_GW_TestContainer_VFL_detector1   # job name
#SBATCH --time=01:00:00                         # dd-hh:mm:ss for the job

#SBATCH -e MMA_GW_TestContainer_VFL_detector1-err-%j.log
#SBATCH -o MMA_GW_TestContainer_VFL_detector1-out-%j.log

#SBATCH --constraint="scratch"

#SBATCH --account=<charging account>
#SBATCH --mail-user=<email-id>
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options


# Load necessary modules
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda deactivate
conda deactivate  # just making sure
module purge
module reset  # load the default Delta modules

module load anaconda3_gpu
module list

# Change directory to the cloned repo
cd <path to cloned repo>

# Start proxystore endpoint on the node (compute/login) where Hanford detector will run
#proxystore-endpoint start mma_gw_detector1

apptainer exec --nv \
  MMA_GW_Inference_miniapp.sif \
  python /app/examples/octopus/run_detector.py --config <absolute path to FL detector0 config file>/detector1.yaml

