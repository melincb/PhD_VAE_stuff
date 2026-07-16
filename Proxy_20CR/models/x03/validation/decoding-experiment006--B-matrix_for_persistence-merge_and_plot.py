#!/usr/bin/env python


import os
import sys
import numpy as np
import multiprocessing

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

import random
from scipy.stats import pearsonr

import iris
import IRData.twcr as twcr
import datetime
from datetime import date, timedelta
start = datetime.datetime.now()

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=1020)
parser.add_argument("--compute_differences", help='Go through all dates and compute differences in latent space for consecutive dates', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--compute_B_matrix", help='Compute B matrix from differences in latent space for consecutive dates', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--plot", help='Plot B matrix', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--parallel_processes", help="Number of CPUs used fo computing differences", type=int, default=4, required=False)
parser.add_argument("--start_year", type=int, required=False, default=2015)
parser.add_argument("--start_month", type=int, required=False, default=1)
parser.add_argument("--start_day", type=int, required=False, default=1)
parser.add_argument("--end_year", type=int, required=False, default=2018)
parser.add_argument("--end_month", type=int, required=False, default=12)
parser.add_argument("--end_day", type=int, required=False, default=31)
args = parser.parse_args()

# Functions for plotting
sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_ERA5_comparison import get_land_mask
from plot_ERA5_comparison import plot_Earth
from plot_ERA5_comparison import plot_colourbar


from ERA5_load import ERA5_load_T850
from ERA5_load import ERA5_load_T850_climatology
from ERA5_load import ERA5_load_T850_variability_climatology
from ERA5_load import ERA5_roll_longitude

import gc

# If computing the differences, set these to dates of interest.
# If only computing the B-matrix, set these in a way that their pkl file already exists and
# that the dates for B-matrix are a subset of these dates.

start_date_B_matrix = date(args.start_year, args.start_month, args.start_day)#date(2015, 1, 1), please start with the first day of the month
end_date_B_matrix = date(args.end_year, args.end_month, args.end_day)#date(2018, 12, 31)

nm = start_date_B_matrix.replace(day=28) + timedelta(days=4)
final_day_of_statring_month = nm - timedelta(days=nm.day)
first_B, len_for_first_B = pickle.load(open(
                f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_epoch={args.epoch}_' + start_date_B_matrix.strftime(
                    '%Y-%m-%d') + '_to_' + final_day_of_statring_month.strftime('%Y-%m-%d') + '.pkl', 'rb'))
previous_B, len_for_previous_B = first_B, len_for_first_B

starting_date_of_current_month = final_day_of_statring_month + timedelta(days=1)
while starting_date_of_current_month < end_date_B_matrix:
    print(starting_date_of_current_month)
    final_day_of_current_month = (starting_date_of_current_month.replace(day=28) + timedelta(days=4)) - timedelta(days=(starting_date_of_current_month.replace(day=28) + timedelta(days=4)).day)
    addon_to_B, len_for_addon_to_B = pickle.load(open(
        f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_epoch={args.epoch}_' + starting_date_of_current_month.strftime(
            '%Y-%m-%d') + '_to_' + final_day_of_current_month.strftime('%Y-%m-%d') + '.pkl', 'rb'))
    previous_B = (previous_B*len_for_previous_B + addon_to_B*len_for_addon_to_B) / (len_for_previous_B + len_for_addon_to_B)  #Weighted average
    len_for_previous_B = len_for_previous_B + len_for_addon_to_B

    starting_date_of_current_month = final_day_of_current_month + timedelta(days=1)   # First day of next month


pickle.dump(previous_B, open(
            f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_epoch={args.epoch}_' + start_date_B_matrix.strftime(
                '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pkl', 'wb'))


