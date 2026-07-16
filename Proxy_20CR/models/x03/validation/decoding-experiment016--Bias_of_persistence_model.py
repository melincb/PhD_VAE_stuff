#!/usr/bin/env python

import os
import sys
import numpy as np
import scipy

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

import random

import iris
import datetime
start = datetime.datetime.now()

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=1020)
parser.add_argument("--compute", help='Compute differences in latent space', default=False, action=argparse.BooleanOptionalAction)
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




start_date = datetime.date(2015, 1, 2)
end_date = datetime.date(2018, 12, 31)
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)



if args.compute:
    # Set up the model and load the weights at the chosen epoch
    sys.path.append("%s/.." % os.path.dirname(__file__))
    from autoencoderModel import DCVAE

    means, successes = [], []
    epochs = []
    constant = 1.0

    matplotlib.rcParams.update({"font.size": 16})

    latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % (1000 * (args.epoch // 1000))]
    autoencoder = DCVAE(latent_dim=latent_dim)

    weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
        args.epoch,
    )
    load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
    # Check the load worked
    devn = load_status.assert_existing_objects_matched()

    autoencoder.decoder.trainable = False
    for layer in autoencoder.decoder.layers:
        layer.trainable = False
    autoencoder.decoder.compile()

    if args.epoch == 1020:
        B_matrix = pickle.load(
            open(
                '../validation/decoding_experiment006---B-matrix_for_persistence-data/B_matrix_2015-01-01_to_2018-12-31.pkl',
                'rb'))
    else:
        print('YOU NEED TO SET A DIFFERENT B-matrix!')
        print(f'YOUR EPOCH IS {args.epoch} AND NOT 1020!')
        raise AttributeError  # YOU NEED TO SET A DIFFERENT B-matrix BECAUSE YOUR EPOCH IS NOT 1020!



    start_date_minus_1_day = start_date - datetime.timedelta(days=1)
    t = ERA5_load_T850(start_date_minus_1_day.year, start_date_minus_1_day.month, start_date_minus_1_day.day)
    c = ERA5_load_T850_climatology(start_date_minus_1_day.year, start_date_minus_1_day.month, start_date_minus_1_day.day)
    vc = ERA5_load_T850_variability_climatology(start_date_minus_1_day.year, start_date_minus_1_day.month, start_date_minus_1_day.day)
    t = t - c
    t = t / vc
    t = ERA5_roll_longitude(t)
    t_in = tf.convert_to_tensor(t.data, np.float32)
    t_in = tf.reshape(t_in, [1, 720, 1440, 1])
    vc = ERA5_roll_longitude(vc)
    vc = vc.data

    latent_mean_old, latent_logvar_old = autoencoder.encode(t_in)
    latent_diffs = []


    for date in daterange(start_date, end_date):
        if date.day == 1:
            print(date)
        t = ERA5_load_T850(date.year, date.month, date.day)
        c = ERA5_load_T850_climatology(date.year, date.month, date.day)
        vc = ERA5_load_T850_variability_climatology(date.year, date.month, date.day)
        t = t - c
        t = t / vc
        t = ERA5_roll_longitude(t)
        t_in = tf.convert_to_tensor(t.data, np.float32)
        t_in = tf.reshape(t_in, [1, 720, 1440, 1])
        vc = ERA5_roll_longitude(vc)
        vc = vc.data

        latent_mean_new, latent_logvar_new = autoencoder.encode(t_in)
        latent_diff = latent_mean_new.numpy() - latent_mean_old.numpy()

        #print(np.squeeze(latent_diff)[0:5])

        latent_diffs.append(np.squeeze(latent_diff))
        latent_mean_old = latent_mean_new#.copy()

    pickle.dump(latent_diffs, open(f'decoding_experiment016--Bias_of_persistence_model-data/latent_diffs_{start_date.year:04d}_{start_date.month:02d}_{start_date.day:02d}_to_{end_date.year:04d}_{end_date.month:02d}_{end_date.day:02d}.pkl', 'wb'))

latent_diffs = pickle.load(open(f'decoding_experiment016--Bias_of_persistence_model-data/latent_diffs_{start_date.year:04d}_{start_date.month:02d}_{start_date.day:02d}_to_{end_date.year:04d}_{end_date.month:02d}_{end_date.day:02d}.pkl', 'rb'))
N_day_avg = 90#15
moving_average = []
eta = []
for i in range(N_day_avg//2, len(latent_diffs) - N_day_avg//2):
    moving_average.append(np.mean(latent_diffs[i-N_day_avg//2:i+N_day_avg//2+1], axis=0))
    eta.append(latent_diffs[i] - moving_average[-1])

print(np.shape(moving_average), np.shape(latent_diffs))

indices = []
i = N_day_avg//2
tick_dates = [datetime.date(2015, 7, 1), datetime.date(2016, 7, 1), datetime.date(2017, 7, 1), datetime.date(2018, 7, 1)]
for date in daterange(start_date, end_date):
    if date in tick_dates:
        indices.append(i)
    i += 1

N_plot = 5
for i in range(N_plot):
    plt.subplot(N_plot, 1, i+1)
    plt.plot(np.array(moving_average)[:, i])
    plt.ylabel(f'idx {i}')
    #plt.xticks([], [])
    plt.xticks(indices, tick_dates)
    plt.grid(linestyle=':', linewidth=0.6, axis='x')

plt.tight_layout()
plt.savefig(f'decoding_experiment016--Bias_of_persistence_model-figures/latent_diffs_{start_date.year:04d}_{start_date.month:02d}_{start_date.day:02d}_to_{end_date.year:04d}_{end_date.month:02d}_{end_date.day:02d}_Ndays={N_day_avg}.jpg', dpi=300)

plt.cla()
plt.clf()
plt.scatter([i for i in range(len(np.mean(eta, axis=0)))], np.mean(eta, axis=0))
plt.tight_layout()
plt.savefig(f'decoding_experiment016--Bias_of_persistence_model-figures/latent_diffs_eta_{start_date.year:04d}_{start_date.month:02d}_{start_date.day:02d}_to_{end_date.year:04d}_{end_date.month:02d}_{end_date.day:02d}_Ndays={N_day_avg}.jpg', dpi=300)

plt.cla()
plt.clf()
print(np.shape(eta))
plt.hist(np.array(eta)[:,0], bins=40)
plt.savefig(f'decoding_experiment016--Bias_of_persistence_model-figures/latent_diffs_eta_hist_{start_date.year:04d}_{start_date.month:02d}_{start_date.day:02d}_to_{end_date.year:04d}_{end_date.month:02d}_{end_date.day:02d}_Ndays={N_day_avg}.jpg', dpi=300)


plt.cla()
plt.clf()
plt.hist(np.array(moving_average)[:, 0], bins=40)
plt.savefig(f'decoding_experiment016--Bias_of_persistence_model-figures/latent_diffs_hist_{start_date.year:04d}_{start_date.month:02d}_{start_date.day:02d}_to_{end_date.year:04d}_{end_date.month:02d}_{end_date.day:02d}_Ndays={N_day_avg}.jpg', dpi=300)

for i in range(6):
    plt.cla()
    plt.clf()
    matplotlib.rcParams.update({"font.size": 16})
    fig, ax1 = plt.subplots()
    nbin = 52
    bins = np.linspace(-2.55, 2.55, nbin)
    c0 = 'C0'
    ax1.hist(np.array(eta)[:,i], bins=bins, color=c0, alpha=0.5, density=True)
    ax1.tick_params(axis='y', labelcolor=c0)
    ax1.set_yticks(ticks=nbin/(max(bins) - min(bins)) * np.array([0.00, 0.05, 0.1]), labels=['0', '5', '10'])
    ax1.set_ylabel(r'Percentage ($\eta$)', color=c0)
    ax1.set_xlabel('Value')
    c1 = 'C1'
    ax2 = ax1.twinx()
    ax2.hist(np.array(moving_average)[:, i], bins=bins, color=c1, alpha=0.5, density=True)
    ax2.tick_params(axis='y', labelcolor=c1)
    ax2.set_yticks(ticks=nbin/(max(bins) - min(bins)) * np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), labels=['0', '20', '40', '60', '80', '100'])
    ax2.set_ylabel(r'Percentage ($b$)', color=c1)
    ax1.set_title(f'idx = {i}')
    plt.tight_layout()
    plt.savefig(f'decoding_experiment016--Bias_of_persistence_model-figures/latent_diffs_hist_combo_{start_date.year:04d}_{start_date.month:02d}_{start_date.day:02d}_to_{end_date.year:04d}_{end_date.month:02d}_{end_date.day:02d}_Ndays={N_day_avg}_idx={i}.jpg', dpi=300)


# all_gp_shapiro_p_mean, all_latent_shapiro_p_mean = pickle.load(open('decoding_experiment015--Gaussianity_of_ERA5_ensemble_members-data/shapiro_p_gp_and_latent.pkl', 'rb'))
# plt.scatter([i for i in dates], all_gp_shapiro_p_mean, marker='s', color='r', label='Original ens. members (grid point space)')
# plt.scatter([i for i in dates], all_latent_shapiro_p_mean, marker='*', color='b', label='Encoded ens. members (latent space)')
# plt.figure(figsize=(12, 5))
# plt.scatter([i for i in range(len(dates))], all_gp_shapiro_p_mean, marker='s', color='r', label='Original ens. members (grid point space)')
# plt.scatter([i for i in range(len(dates))], all_latent_shapiro_p_mean, marker='*', color='b', label='Encoded ens. members (latent space)')
# plt.xticks(ticks=[i for i in range(len(dates))], labels=[d.strftime("%Y-%m-%d") for d in dates], rotation=90)
# plt.grid(linestyle=':', linewidth=0.6)
# plt.legend(loc='lower right')
# plt.ylabel('p-value from Shapiro-Wilk test')
# plt.tight_layout()
# plt.savefig('decoding_experiment015--Gaussianity_of_ERA5_ensemble_members-data/shapiro_p_gp_and_latent.jpg', dpi=300)
#
# print(np.mean(all_gp_shapiro_p_mean), np.std(all_gp_shapiro_p_mean), np.mean(all_latent_shapiro_p_mean), np.std(all_latent_shapiro_p_mean))