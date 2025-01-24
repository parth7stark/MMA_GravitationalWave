#!/bin/bash
#SBATCH --mem=240g                              # required number of memory
#SBATCH --nodes=1                               # nodes required for whole simulation <-- determine from architecture and distribution of GPUs/CPUs

#SBATCH --cpus-per-task=64                       # CPUs for each task/client
##SBATCH --gpus-per-task=1
##SBATCH --ntasks-per-node=1

#SBATCH --partition=cpu                # server doesn't need gpu <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8

##SBATCH --gpus-per-node=1                       # number of GPUs you want to use on 1 node
##SBATCH --gpu-bind=none

#SBATCH --job-name=MMA_GW_TestContainer_VFL_client2_CPU    # job name
#SBATCH --time=01:00:00                         # dd-hh:mm:ss for the job

#SBATCH -e MMA_GW_TestContainer_VFL_client2_CPU-err-%j.log
#SBATCH -o MMA_GW_TestContainer_VFL_client2_CPU-out-%j.log

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

# source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
# conda activate /u/parthpatel7173/.conda/envs/vfl1


# cd /u/parthpatel7173/GW_VerticalFL/examples

# Start the server -- automatically set server IP and port in config and print it out

# python3 ./grpc/run_vfl_client.py --config ./configs/vfl/client2.yaml

cd /scratch/bcbw/parthpatel7173/MMA_GravitationalWave

apptainer exec --nv \
  GW_miniapp_delta.sif \
  python /app/examples/octopus/run_detector.py --config /scratch/bcbw/parthpatel7173/MMA_GravitationalWave/examples/configs/client2_container.yaml

