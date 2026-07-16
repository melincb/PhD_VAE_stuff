#!/usr/bin/env python

import os
import sys
import numpy as np

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
parser.add_argument("--compute", help='Compute std of background', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=1020)
parser.add_argument("--ensemble", help="Ensemble size", type=int, required=False, default=150)
parser.add_argument("--ensembles", help="Number of ensembles", type=int, required=False, default=100)
parser.add_argument("--year", type=int, required=False, default=2019)
parser.add_argument("--month", type=int, required=False, default=4)
parser.add_argument("--day", type=int, required=False, default=15)
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


# Set up the model and load the weights at the chosen epoch
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

means, successes= [], []
epochs = []
constant = 1.0

matplotlib.rcParams.update({"font.size": 16})


autoencoder = DCVAE()

weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
    args.epoch,
)
#print('Uploading weights from', weights_dir)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
# Check the load worked
devn = load_status.assert_existing_objects_matched()
#print('Weights loaded, objects match')

# We are using it in inference mode
# (I'm not at all sure this actually works)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()
#print('Autoencoder compiled')

if args.epoch == 1020:
    B_matrix = pickle.load(
        open('../validation/decoding_experiment006---B-matrix_for_persistence-data/B_matrix_2015-01-01_to_2018-12-31.pkl',
             'rb'))
else:
    print('YOU NEED TO SET A DIFFERENT B-matrix!')
    print(f'YOUR EPOCH IS {args.epoch} AND NOT 1020!')
    raise AttributeError #YOU NEED TO SET A DIFFERENT B-matrix BECAUSE YOUR EPOCH IS NOT 1020!

if args.compute:
    dyear = [0,1]
    for d in dyear:
        date = datetime.datetime(year=args.year+d, month=args.month, day=args.day)

        t = ERA5_load_T850(args.year+d, args.month, args.day)
        t_orig = t.copy()
        c = ERA5_load_T850_climatology(args.year+d, args.month, args.day)
        vc = ERA5_load_T850_variability_climatology(args.year+d, args.month, args.day)
        t = t - c
        t = t / vc
        t = ERA5_roll_longitude(t)
        # t = ERA5_trim(t)
        t_in = tf.convert_to_tensor(t.data, np.float32)
        t_in = tf.reshape(t_in, [1, 720, 1440, 1])

        vc = ERA5_roll_longitude(vc)
        vc = vc.data


        latent_mean_orig, latent_logvar_orig = autoencoder.encode(t_in)

        stds = []

        for i in range(args.ensembles):
            print(args.year + d, i)
            latent_samples = tf.Variable(tf.random.normal(mean=latent_mean_orig, stddev=np.sqrt(np.diagonal(B_matrix)), shape=(args.ensemble, autoencoder.latent_dim)))
            decoded = autoencoder.decode(latent_samples)
            print('Decoded')

            e_std = tf.math.reduce_std(decoded, axis=0)
            stds.append(tf.squeeze(e_std).numpy() * vc)

        pickle.dump(np.array(stds, dtype='float16'), open(f'decoding_experiment009--Flow_dependent_background_ensemble_size-data/de009--{args.year+d}-{args.month}-{args.day}_ens{args.ensemble}_enss{args.ensembles}.pkl', 'wb'))


stds0 = pickle.load(open(f'decoding_experiment009--Flow_dependent_background_ensemble_size-data/de009--{args.year}-{args.month}-{args.day}_ens{args.ensemble}_enss{args.ensembles}.pkl', 'rb'))
stds1 = pickle.load(open(f'decoding_experiment009--Flow_dependent_background_ensemble_size-data/de009--{args.year+1}-{args.month}-{args.day}_ens{args.ensemble}_enss{args.ensembles}.pkl', 'rb'))

mean_of_stds0 = np.mean(stds0, axis=0)
std_of_stds0 = np.std(stds0, axis=0)
mean_of_stds1 = np.mean(stds1, axis=0)
std_of_stds1 = np.std(stds1, axis=0)

std_of_stds_max = np.maximum(std_of_stds0, std_of_stds1)

phi = np.array([[lat*np.pi/180 for lon in range(1440)] for lat in np.linspace(-89.875, 89.875, 720, endpoint=True)])
cosphi = np.cos(phi)
cosphi = cosphi.astype(np.float32)
weighted_successful_max = np.sum(cosphi)

print(std_of_stds_max)
print(np.abs(mean_of_stds0 - mean_of_stds1))

weighted_successful = np.sum(np.where(np.abs(mean_of_stds0 - mean_of_stds1) > std_of_stds_max, 1, 0) * cosphi)

weighted_successes = weighted_successful / weighted_successful_max * 100

print(weighted_successes)



fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lm = get_land_mask()
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.contourf(lons, lats, np.abs(mean_of_stds0 - mean_of_stds1) / std_of_stds_max, transform=ccrs.PlateCarree(),
                    levels = [0,1,2,3],
                    cmap='viridis', extend='max')
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='max', label=r'abs{$M(\sigma_{b}^{%02d}) - M(\sigma_{b}^{%02d})$} / max{$S(\sigma_{b}^{%02d})$, $S(\sigma_{b}^{%02d})$}' % (args.year%1000, (args.year+1)%1000, args.year%1000, (args.year+1)%1000))
ax2.set_title('Normalised difference', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    'decoding_experiment009--Flow_dependent_background_ensemble_size-figures/de009--Flow_dependent_background_ensemble_size_%02d-%02d_%04d_and_%04d-ens%d_enss%d_ratio'
    % (args.month, args.day, args.year, args.year+1, args.ensemble, args.ensembles) + '.jpg', dpi=300)

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lm = get_land_mask()
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, mean_of_stds0 - mean_of_stds1, transform=ccrs.PlateCarree(),
                    cmap='PuOr_r', vmin=-2, vmax=2)
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05,  extend='both', label=r'$M(\sigma_{b}^{%02d}) - M(\sigma_{b}^{%02d})$ [$\degree$C]' % (args.year%1000, (args.year+1)%1000))
ax2.set_title('Difference', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    'decoding_experiment009--Flow_dependent_background_ensemble_size-figures/de009--Flow_dependent_background_ensemble_size_%02d-%02d_%04d_and_%04d-ens%d_enss%d_diff'
    % (args.month, args.day, args.year, args.year+1, args.ensemble, args.ensembles) + '.jpg', dpi=300)

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lm = get_land_mask()
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, mean_of_stds0 / mean_of_stds1, transform=ccrs.PlateCarree(),
                    cmap='PiYG_r', vmin=0.5, vmax=1.5)
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05,  extend='both', label=r'$M(\sigma_{b}^{%02d}) \,/ \,M(\sigma_{b}^{%02d})$' % (args.year%1000, (args.year+1)%1000))
ax2.set_title('Ratio', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    'decoding_experiment009--Flow_dependent_background_ensemble_size-figures/de009--Flow_dependent_background_ensemble_size_%02d-%02d_%04d_and_%04d-ens%d_enss%d_true_ratio'
    % (args.month, args.day, args.year, args.year+1, args.ensemble, args.ensembles) + '.jpg', dpi=300)



fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lm = get_land_mask()
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, mean_of_stds0, transform=ccrs.PlateCarree(),
                    cmap='terrain_r', vmin=0, vmax=5)
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05,  extend='max', label=r'mean of $\sigma_{b}^{%04d}$ [$\degree$C]' % (args.year))
ax2.set_title('Mean std of backg. for %04d-%02d-%02d' % (args.year, args.month, args.day), y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    'decoding_experiment009--Flow_dependent_background_ensemble_size-figures/de009--Flow_dependent_background_ensemble_size_%02d-%02d_%04d_and_%04d-ens%d_enss%d_only_prev_year'
    % (args.month, args.day, args.year, args.year+1, args.ensemble, args.ensemble+1) + '.jpg', dpi=300)


fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lm = get_land_mask()
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, mean_of_stds1, transform=ccrs.PlateCarree(),
                    cmap='terrain_r', vmin=0, vmax=5)
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05,  extend='max', label=r'mean of $\sigma_{b}^{%04d}$ [$\degree$C]' % (args.year+1))
ax2.set_title('Mean std of backg. for %04d-%02d-%02d' % (args.year+1, args.month, args.day), y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    'decoding_experiment009--Flow_dependent_background_ensemble_size-figures/de009--Flow_dependent_background_ensemble_size_%02d-%02d_%04d_and_%04d-ens%d_enss%d_only_next_year'
    % (args.month, args.day, args.year, args.year+1, args.ensemble, args.ensemble+1) + '.jpg', dpi=300)