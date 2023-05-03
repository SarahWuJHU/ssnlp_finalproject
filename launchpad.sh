#!/bin/bash
#SBATCH --qos=qos_gpu
#SBATCH -A danielk80_gpu 
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=2:00:00
#SBATCH --job-name="swu82 Final Project"

module load anaconda
#init virtual environment if needed
# conda create -n nlp3 python=3.8

conda activate nlp3 # open the Python environment

# runs your code
srun python model.py