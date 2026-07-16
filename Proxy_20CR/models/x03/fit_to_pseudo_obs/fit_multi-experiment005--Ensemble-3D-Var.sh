#!/bin/bash
echo -e "You've entered the script for Ensemble 3D-Var."
echo -e "You should've set the following variables:"
echo -e "\tDo you want to assimilate data with Ensemble 3D-Var (i.e. to run the computation)? y/n (variable DoIPrepare)"	#DoIPrepare
echo -e "\tDo you want to plot the results? y/n (variable DoIPlot)"	#DoIPlot

DoIPrepare=$1
DoIPlot=$2

echo -e "The variables you've set are:"
echo -e "\tDoIPrepare: $DoIPrepare"
echo -e "\tDoIPlot: $DoIPlot"

echo -e "\nSleeping 5 seconds (might be a proper moment to kill the program...)"
sleep 5

settings=$3
#settings="--epoch=1020 --minimization_learning_rate=0.1 --cpus=14 --ensemble=150 --obs_std=1.0 --diagonal_B --plot_singles"
#"--epoch=1020 --minimization_learning_rate=0.01 --cpus=30 --ensemble=150 --obs_increment=3.0 --custom_addon=singobs_Nebraska+3K --obs_std=1.0 --singobs_lon=-100.0 --singobs_lat=42.0 --diagonal_B --plot_singles"
#"--epoch=1020 --minimization_learning_rate=0.01 --cpus=14 --ensemble=150 --obs_increment=3.0 --custom_addon=singobs_East_Pacific+3K --obs_std=1.0 --singobs_lon=-85.0 --singobs_lat=0.0 --diagonal_B --plot_singles"
#"--epoch=1020 --minimization_learning_rate=0.01 --cpus=14 --ensemble=150 --obs_increment=3.0 --custom_addon=singobs_Ljubljana+3K --obs_std=1.0 --singobs_lon=14.506 --singobs_lat=46.057 --diagonal_B --plot_singles"
#"--epoch=1020 --minimization_learning_rate=0.01 --cpus=14 --ensemble=150 --obs_increment=3.0 --custom_addon=singobs_CAR+3K --obs_std=1.0 --singobs_lon=21.000 --singobs_lat=7.000 --diagonal_B" #--plot_singles"
#"--epoch=1020 --minimization_learning_rate=0.1 --cpus=14 --ensemble=150 --obs_std=1.0 --diagonal_B" # --plot_singles
#"--epoch=1020 --minimization_learning_rate=0.01 --cpus=14 --ensemble=150 --obs_increment=0.0 --custom_addon=singobs_Ljubljana_from_true_field --obs_std=1.0 --singobs_lon=14.506 --singobs_lat=46.057 --diagonal_B --plot_singles"
#"--epoch=1020 --minimization_learning_rate=0.1 --cpus=14 --ensemble=150 --obs_std=1.0 --diagonal_B --plot_singles"
#"--epoch=1020 --minimization_learning_rate=0.01 --cpus=14 --ensemble=150 --obs_increment=3.0 --custom_addon=singobs_Billings+3K --obs_std=1.0 --singobs_lon=-108.5 --singobs_lat=45.8 --diagonal_B --plot_singles --month=07"
#--diagonal_B 
prepare="--compute"
plot="--plot"

echo $(date '+%Y-%m-%d %H:%M:%S')

case $DoIPrepare in
	[Yy] )	echo -e "\n\nMoving previous algorithm inputs (if they exist)"
		if [ -e "fit_multi-experiment005--Ensemble-3D-Var-data/algorithm_inputs.pkl" ]; then
			mv fit_multi-experiment005--Ensemble-3D-Var-data/algorithm_inputs.pkl fit_multi-experiment005--Ensemble-3D-Var-data/algorithm_inputs_previous.pkl
		fi
		echo -e "\n\nPreparing data by running the following command:"
		echo -e "python fit_multi-experiment005--Ensemble-3D-Var--prepare_or_plot.py $settings $prepare"
		sleep 1
		python fit_multi-experiment005--Ensemble-3D-Var--prepare_or_plot.py $settings $prepare
		echo -e "\n\nExecuting the Ensemble 3D-Var algorithm by running the following command:"
		echo "python fit_multi-experiment005-Ensemble-3D-Var--algorithm.py"
		sleep 1
		python fit_multi-experiment005--Ensemble-3D-Var--algorithm.py
		echo -e "\n\nComputation ended!"
		;;
	*)	;;
esac

case $DoIPlot in
	[Yy] )	echo -e "\n\nPlotting results by running the following command:"
		echo "python fit_multi-experiment005--Ensemble-3D-Var--prepare_or_plot.py $settings $plot"
		sleep 1
		python fit_multi-experiment005--Ensemble-3D-Var--prepare_or_plot.py $settings $plot
		echo -e "\n\nPlotting ended!"
		;;
	*)	;;
esac

echo "Bash script finished!"
echo $(date '+%Y-%m-%d %H:%M:%S')
