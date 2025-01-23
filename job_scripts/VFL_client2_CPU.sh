#!/bin/bash
#SBATCH --mem=240g                              # required number of memory
#SBATCH --nodes=1                               # nodes required for whole simulation <-- determine from architecture and distribution of GPUs/CPUs

#SBATCH --cpus-per-task=64                       # CPUs for each task/client

#SBATCH --partition=cpu                  # server doesn't need gpu <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8


#SBATCH --job-name=MMA_GW_Test_VFL_client2    # job name
#SBATCH --time=01:00:00                         # dd-hh:mm:ss for the job

#SBATCH -e MMA_GW_Test_VFL_client2-err-%j.log
#SBATCH -o MMA_GW_Test_VFL_client2-out-%j.log

#SBATCH --constraint="scratch"

#SBATCH --account=bbjo-delta-cpu
#SBATCH --mail-user=pp32@illinois.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options



source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda deactivate
conda deactivate  # just making sure
module purge
module reset  # load the default Delta modules

module load anaconda3_gpu
module list

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate /u/parthpatel7173/.conda/envs/mma_gw


cd /scratch/bcbw/parthpatel7173/MMA_GravitationalWave

# Start detector 1 -- update config before starting

python3 ./examples/octopus/run_detector.py --config ./examples/configs/client2.yaml


