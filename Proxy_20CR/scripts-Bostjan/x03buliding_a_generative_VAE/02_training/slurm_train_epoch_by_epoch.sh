#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:59
#SBATCH --mem=70G
#SBATCH --cpus-per-task=2
#SBATCH --partition=rude
#SBATCH --qos=rude
#SBATCH --error=train_epoch_by_epoch.%J.err
#SBATCH --output=train_epoch_by_epoch.%J.out
#SBATCH --job-name=bVAE4single

srun bash train_epoch_by_epoch.sh
