#!/bin/bash
# Ne pozabi v fit_multi-experiment005--Ensemble-3D-Var.sh odkomentirati vrstice settings=$3 in zakomentirati naslednje!
startDay=15
startMonth=4
startYear=2019
cyclingStartingPoint="$startYear-$startMonth-$startDay"

ensemble=150
obs_std=1.0
mlr=0.1


starting_date_object=$(date -d "$cyclingStartingPoint" "+%Y-%m-%d")

cycles=30
echo "Cycle 1: $cyclingStartingPoint"
bash fit_multi-experiment005--Ensemble-3D-Var.sh n y "--minimization_learning_rate=$mlr --cpus=30 --ensemble=$ensemble --obs_std=$obs_std --diagonal_B --cycling_starting_point=$cyclingStartingPoint --year=$startYear --month=$startMonth --day=$startDay"

for (( cycle=2; cycle<=$cycles; cycle++ ))
do
	new_date=$(date -d "$starting_date_object + $((cycle - 1)) days" "+%Y-%m-%d")
	old_date=$(date -d "$starting_date_object + $((cycle - 2)) days" "+%Y-%m-%d")
	echo "Cycle $cycle: $new_date"
	sleep 3
	year=$(date -d "$new_date" "+%Y")
	month=$(date -d "$new_date" "+%-m")
	day=$(date -d "$new_date" "+%-d")
	old_year=$(date -d "$old_date" "+%Y")
	old_month=$(date -d "$old_date" "+%-m")
	old_day=$(date -d "$old_date" "+%-d")
	background_latent_mean_field_file="fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_$old_year-$old_month-$((old_day))_epoch=1020_obs_std=$obs_std""_res=4.0_ensemble=$((ensemble))minimization_lr=$mlr""_diagonal_B_cycling_from_$cyclingStartingPoint"
	bash fit_multi-experiment005--Ensemble-3D-Var.sh n y "--minimization_learning_rate=$mlr --cpus=30 --ensemble=$ensemble --obs_std=$obs_std --diagonal_B --cycling_starting_point=$cyclingStartingPoint --year=$year --month=$month --day=$day --background_latent_mean_field_file=$background_latent_mean_field_file"
done



