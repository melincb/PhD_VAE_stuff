#!/bin/bash
echo "Sourcing ProxyR env"
source BP/anaconda3/bin/activate ProxyR
echo "Sourced ProxyR env"


experiment="x03"
startingepoch=$1
endingepoch=$2

echo -e "Computing the loss for the entire training set\n(=Running models/$experiment/models_by_epochs/full_history.py)\nStarting epoch = $startingepoch\nEnding epoch = $endingepoch"

for trainingSection in {0..9}
do
	echo "Training section $trainingSection - generating slurm script"
	python generate_slurm_script.py --start=$startingepoch --end=$endingepoch --training_section=$trainingSection
	echo "Training section $trainingSection - running slurm script"
	sbatch slurm_${startingepoch}_${endingepoch}_ts$trainingSection.sh
done

echo "Done!"
echo $(date '+%Y-%m-%d %H:%M:%S')


