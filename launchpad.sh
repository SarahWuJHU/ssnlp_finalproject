#!/bin/bash
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name="Final Project"

module load anaconda
#init virtual environment if needed
# conda create -n nlp3 python=3.8

conda activate nlp3 # open the Python environment

# runs your code
srun python model.py