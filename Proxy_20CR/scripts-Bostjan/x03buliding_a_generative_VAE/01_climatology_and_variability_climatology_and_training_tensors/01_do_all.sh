#!/bin/bash
#Paths have been blurred for security reasons
echo "Sourcing ProxyR"
source BLURRED_PATH/anaconda3/bin/activate ProxyR
echo "Sourced ProxyR"

echo $(date)

echo "Creating slurm scripts for climatology, variability climatology and training tensors"
for month in {1..12}
do
	echo "Month $month"
	python generate_slurm_script.py --month=$month
done

echo "Running slurm scripts for climatology, variability climatology and training tensors"
for month in {1..12}
do
	echo "Month $month"
	sbatch slurm_$month.sh
done

