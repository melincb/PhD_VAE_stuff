#!/usr/bin/env python
import gc
# Find a point in latent space that maximises the fit to a set of pseudo-obs,
#  and plot the fitted state.
# Make multiple fits and plot the ensemble.

import os
import sys
import time

import numpy as np
import scipy.stats

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear


import random

import iris
import IRData.twcr as twcr
import datetime

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import argparse
import pickle


# start = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=1020)
# parser.add_argument(
#     "--ensemble", help="No. of ensemble members", type=int, required=False, default=50
# )
parser.add_argument("--year", help="Year", type=int, required=False, default=2019)
parser.add_argument(
    "--month", help="Integer month", type=int, required=False, default=4
)
parser.add_argument("--day", help="Day of month", type=int, required=False, default=15)
# parser.add_argument("--oyear", help="Year", type=int, required=False)
# parser.add_argument("--omonth", help="Integer month", type=int, required=False)
# parser.add_argument("--oday", help="Day of month", type=int, required=False)
# parser.add_argument(
#     "--osize", help="Obs. point size", type=float, required=False, default=1.0
# )
# parser.add_argument('--compute', help="Compute assimilation", default=False, action=argparse.BooleanOptionalAction)  #In order not to compute: --no-compute
# parser.add_argument('--plot', help="Plot", default=False, action=argparse.BooleanOptionalAction) #In order not to plot: --no-plot
# parser.add_argument('--std_first_multiplier', help="Multiplier of std of first guess", type=float, required=False, default=1.0)
# parser.add_argument('--obs_std', help="Standard deviation of pseudo observations, degree C", type=float, required=False, default=0.0)
# parser.add_argument('--minimization_learning_rate', help='Learning rate for ADAM optimizer when performing minimization in latent space', type=float, required=False, default=0.01)
# parser.add_argument('--adaptive_lr', help='Whether the learning rate for ADAM optimizer when performing minimization in latent space decreases if loss is on plateau or not', default=True, action=argparse.BooleanOptionalAction)
# parser.add_argument('--perfect_obs', help='Perfect observations', default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument('--perfect_first', help='Perfect first guess', default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument('--save_as_pdf', help='Save final figure also in pdf format', default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument('--custom_addon', type=str, default='', required=False)
# parser.add_argument('--cpus', type=int, default=1, required=False)
# parser.add_argument('--obs_increment', help="Observation increment for single observation experiment (only works if not 0.0)", type=float, required=False, default=0.0)
# parser.add_argument('--diagonal_B', help='Only use diagonal elements of B-matrix (no correlations between latent elements)', default=False, action=argparse.BooleanOptionalAction)
#
args = parser.parse_args()
# if args.oyear is None:
#     args.oyear = args.year
# if args.omonth is None:
#     args.omonth = args.month
# if args.oday is None:
#     args.oday = args.day

# check_if_in_test_set_path = '%s/Proxy_20CR/datasets/ERA5/daily_T850/regridded_version/x03test/%04d-%02d-%02d.tfd' % (os.getenv("SCRATCH"), args.year, args.month, args.day)
# if os.path.isfile(check_if_in_test_set_path):
#     print('\nChosen date is in the TEST set!\n')
# else:
#     check_if_in_validation_set_path = '%s/Proxy_20CR/datasets/ERA5/daily_T850/regridded_version/x03validation/%04d-%02d-%02d.tfd' % (os.getenv("SCRATCH"), args.year, args.month, args.day)
#     if os.path.isfile(check_if_in_validation_set_path):
#         print('\nChosen date is in the VALIDATION set!\n')
#     else:
#         print('\nChosen date IS NOT in the validation or test set\n')

# Functions for plotting
sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_ERA5_comparison import get_land_mask
from plot_ERA5_comparison import plot_Earth
from plot_ERA5_comparison import plot_colourbar

# Make the input tensor for the specified date
sys.path.append(
    "%s/../../../data/prepare_training_tensors_ERA5_T850" % os.path.dirname(__file__)
)
from ERA5_load import ERA5_load_T850
from ERA5_load import ERA5_load_T850_climatology
from ERA5_load import ERA5_load_T850_variability_climatology
from ERA5_load import ERA5_roll_longitude


# Set up the model and load the weights at the chosen epoch
print('PREPARING VAE!')
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
# Check the load worked
devn = load_status.assert_existing_objects_matched()

# We are using it in inference mode
# (I'm not at all sure this actually works)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()

# min_lr = args.minimization_learning_rate


#
# res = 'custom'#4.0   # distance between observation points (or 'custom')
# double_res = False#True # whether we also add the same grid, but diagonaly shifted
# fake_zeros = False  # if true, the climatological mean is used as the input
# res_str = str(res)
# if double_res:
#     res_str += 'd'


# def log_normal_pdf(sample, mean, logvar, raxis=1):
#     log2pi = tf.math.log(2.0 * np.pi)
#     return tf.reduce_sum(
#         -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
#     )

print('PREPARING DATA!')
this_day = datetime.date(args.year, args.month, args.day)
t = ERA5_load_T850(args.year, args.month, args.day)
c = ERA5_load_T850_climatology(args.year, args.month, args.day)
vc = ERA5_load_T850_variability_climatology(args.year, args.month, args.day)
t = t - c
t = t / vc
t = ERA5_roll_longitude(t)
t_in = tf.convert_to_tensor(t.data, np.float32)
t_in = tf.reshape(t_in, [1, 720, 1440, 1])
vc = ERA5_roll_longitude(vc)
vc = vc.data

# previous_day = this_day - datetime.timedelta(days=1)
# t_previous = ERA5_load_T850(previous_day.year, previous_day.month, previous_day.day)
# c_previous = ERA5_load_T850_climatology(previous_day.year, previous_day.month, previous_day.day)
# vc_previous = ERA5_load_T850_variability_climatology(previous_day.year, previous_day.month, previous_day.day)
# t_previous = t_previous - c_previous
# if fake_zeros:
#     t_previous = t_previous / t_previous * 0
# t_previous = t_previous / vc_previous
# t_previous = ERA5_roll_longitude(t_previous)
# t_previous_in = tf.convert_to_tensor(t_previous.data, np.float32)
# t_previous_in = tf.reshape(t_previous_in, [1, 720, 1440, 1])
# vc_previous = ERA5_roll_longitude(vc_previous)
# vc_previous = vc_previous.data
#
# true_mean, true_logvar = autoencoder.encode(t_in)
# true_mean = true_mean.numpy().reshape(autoencoder.latent_dim)
# true_logvar = true_logvar.numpy().reshape(autoencoder.latent_dim)
# true_std = np.sqrt(np.exp(true_logvar))
# true_half_iqr = true_std * 0.6744  # Bostjan: experimented with scipy.stats.norm.cdf, also in Bronstein (ish)
#
# previous_true_mean, previous_true_logvar = autoencoder.encode(t_previous_in)
# previous_true_mean = previous_true_mean.numpy().reshape(autoencoder.latent_dim)
# previous_true_logvar = previous_true_logvar.numpy().reshape(autoencoder.latent_dim)
# previous_true_std = np.sqrt(np.exp(previous_true_logvar))
# previous_true_half_iqr = previous_true_std * 0.6744




    #file_to_load = f'fit_multi-experiment004--3D-Var_obs_on_regular_grid-data/fe003--Fit_pseudo_obs_on_quasi_regular_grid-data_{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}_obs_std={args.obs_std}_res={res}'
file_to_load1 = 'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_2019-4-15_epoch=1020_obs_std=1.0_res=custom_ensemble=150_diagonal_B_singobs_Ljubljana+3K.pkl' # first analysis pickle file
file_to_load2 = 'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_2019-4-15_epoch=1020_obs_std=1.0_res=custom_ensemble=150_singobs_Ljubljana+3K.pkl' # second analysis pickle file
name_addon = '2019-4-15_epoch=1020_obs_std=1.0_ensemble=150_singobs_Ljubljana+5K__diag_B_vs_full_B'# for the saved figure

# file_to_load1 = 'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_2019-4-15_epoch=1020_obs_std=1.0_res=4.0_ensemble=150minimization_lr=0.05_diagonal_B.pkl'
# file_to_load2 = 'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_2019-4-15_epoch=1020_obs_std=1.0_res=4.0_ensemble=150minimization_lr=0.05.pkl'
# name_addon = '2019-4-15_epoch=1020_obs_std=1.0_res=4.0_ensemble=150__diagonal_B_vs_full_B'# for the saved figure

dict_to_load1 = pickle.load(open(file_to_load1, 'rb'))
latent1 = dict_to_load1['latent']
t_obs1 = dict_to_load1['t_obs']
previous_latents1 = dict_to_load1['previous_latent']
obs_gp_std1 = dict_to_load1['obs_gp_std']
best_loss1 = dict_to_load1['best_loss']
loss1 = dict_to_load1['loss']
logpzs1 = dict_to_load1['logpzs']
comment1 = dict_to_load1['comment']
all_Jo1 = dict_to_load1['all_Jo']
all_gradients1 = dict_to_load1['all_gradients']

lnpy1 = latent1.numpy()


dict_to_load2 = pickle.load(open(file_to_load2, 'rb'))
latent2 = dict_to_load2['latent']
t_obs2 = dict_to_load2['t_obs']
previous_latents2 = dict_to_load2['previous_latent']
obs_gp_std2 = dict_to_load2['obs_gp_std']
best_loss2 = dict_to_load2['best_loss']
loss2 = dict_to_load2['loss']
logpzs2 = dict_to_load2['logpzs']
comment2 = dict_to_load2['comment']
all_Jo2 = dict_to_load2['all_Jo']
all_gradients2 = dict_to_load2['all_gradients']

lnpy2 = latent2.numpy()

# This always loads the same B matrix!
B_matrix = pickle.load(open(
    '../validation/decoding_experiment006---B-matrix_for_persistence-data/B_matrix_2015-01-01_to_2018-12-31.pkl',
    'rb'))

#print(lnpy[:,0])

# Compute standard deviation of the background (propagated to the gridpoint space)
# background_gp1 = tf.squeeze(autoencoder.decode(previous_latents1)) * vc
# background_gp_std1 = tf.math.reduce_std(background_gp1, axis=0).numpy()
# background_gp_mean1 = tf.math.reduce_mean(background_gp1, axis=0).numpy()


# And finally, lets decode the fitted latent space
fitted1 = autoencoder.decode(latent1)
e_mean1 = tf.math.reduce_mean(fitted1, axis=0)
e_std1 = tf.math.reduce_std(fitted1, axis=0)
# print('fitted1')
# e_mean1 = tf.squeeze(tf.math.reduce_mean(fitted1, axis=0)).numpy() * vc
# #print('fitted1', np.shape(e_mean1), e_mean1)
# e_std1 = tf.squeeze(tf.math.reduce_std(fitted1, axis=0)).numpy() * vc
# print('fitted1')
fitted2 = autoencoder.decode(latent2)
e_mean2 = tf.math.reduce_mean(fitted2, axis=0)
e_std2 = tf.math.reduce_std(fitted2, axis=0)
print('fitted2')
print(e_mean1/e_mean2)

print('Plotting!')

fig = Figure(
    figsize=(15, 10),
    dpi=300,
    facecolor="white",
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
matplotlib.rcParams.update({"font.size": 16})

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
ax_global.set_axis_off()
ax_global.autoscale(enable=False)
ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")


plot_left = 0.075
plot_width = 0.85
cbars = 2
tot_plots = 2
height_sum = tot_plots * 0.075 + cbars * 0.015 + 0.005 * (tot_plots + cbars + 1)
dheight_plot = 0.075 / height_sum
dheight_cbar = 0.015 / height_sum
dheight_buffer = 0.005 / height_sum
plot_bottom = 1

lm = get_land_mask()
# 1st (=top) - difference between the analyses means
plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
print([plot_left, plot_bottom, plot_width, dheight_plot])
ax_da = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
ax_da.set_aspect("equal")
ax_da.set_xticks([])
ax_da.set_yticks([])
ax_da.set_xlim(-180, 180)
ax_da.set_ylim(-90, 90)
ax_da.set_ylabel('Difference between analyses' + r' [$\degree$C]')
oda = plot_Earth(
    ax_da,
    tf.squeeze(e_mean1).numpy() * vc - tf.squeeze(e_mean2).numpy() * vc,
    vMin=-2,
    vMax=2,
    land=lm,
    #label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
)


plot_bottom += dheight_plot + dheight_buffer
ax_dacb = fig.add_axes([(1-0.5)/2, plot_bottom, 0.5, dheight_cbar])
plot_colourbar(fig, ax_dacb, oda)
plot_bottom -= dheight_plot + dheight_buffer
print('got 1st')

central_latitude = (1 - t_obs1[0][0][0] / 720) * 180 - 90
central_longitude = t_obs1[0][0][1] / 1440 * 360 - 180
fontsize = 16
transform = ccrs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_mean1).numpy() * vc - tf.squeeze(e_mean2).numpy() * vc,
                    transform=ccrs.PlateCarree(), vmin=-3, vmax=3, cmap='seismic')

ax2.scatter([central_longitude], [central_longitude], c='gold', s=80.0, marker='*', edgecolor='k', linewidth=0.5,
            zorder=10 ** 4)
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, extend='both', label=r'$T_{850}$ [$\degree$C]', ticks=[-3, -2, -1, 0, 1, 2, 3])
ticklabs = cb2.ax.get_yticklabels()
cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
ax2.set_title(r"Analysis difference (diag. B - full B)", y=1.02, fontsize=fontsize)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-compare_analyses-{name_addon}_analysis_difference.jpg', dpi=300
    )
print('saved single fig.')

# 2nd ratio between standard deviations
plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer

ax_dstd = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
print([plot_left, plot_bottom, plot_width, dheight_plot])
ax_dstd.set_aspect("equal")
ax_dstd.set_xticks([])
ax_dstd.set_yticks([])
ax_dstd.set_xlim(-180, 180)
ax_dstd.set_ylim(-90, 90)
ax_dstd.set_ylabel(r'Standard deviations ratio')
odstd = plot_Earth(
    ax_dstd,
    tf.squeeze(e_std1).numpy()/tf.squeeze(e_std2).numpy(),
    vMin=0,
    vMax=2,
    #obs=tf.squeeze(t_obs, [0]).numpy(),
    #o_size=0.5,
    land=lm,
    cmap='PiYG_r'
)

plot_bottom += dheight_plot + dheight_buffer
ax_stdcb = fig.add_axes([(1 - 0.5) / 2, plot_bottom, 0.5, dheight_cbar])
plot_colourbar(fig, ax_stdcb, odstd)
plot_bottom -= dheight_plot + dheight_buffer


central_latitude = (1 - t_obs1[0][0][0] / 720) * 180 - 90
central_longitude = t_obs1[0][0][1] / 1440 * 360 - 180
fontsize = 16
transform = ccrs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_std1).numpy()/tf.squeeze(e_std2).numpy(),
                    transform=ccrs.PlateCarree(), vmin=0.9, vmax=1.1, cmap='PiYG_r')
print('Min', np.amin(tf.squeeze(e_std1).numpy()/tf.squeeze(e_std2).numpy()))
print('Max', np.amax(tf.squeeze(e_std1).numpy()/tf.squeeze(e_std2).numpy()))
ax2.scatter([central_longitude], [central_longitude], c='gold', s=80.0, marker='*', edgecolor='k', linewidth=0.5,
            zorder=10 ** 4)
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, extend='both')
ticklabs = cb2.ax.get_yticklabels()
cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
ax2.set_title(r"Analysis std ratio (diag. B / full B)", y=1.02, fontsize=fontsize)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-compare_analyses-{name_addon}_std_ratio.jpg', dpi=300
    )
print('saved single fig.')


fig.savefig(f'fit_multi-experiment005--Ensemble-3D-Var-figures/fe005--Ensemble-3D-Var-compare_analyses-{name_addon}.jpg', dpi=300)
print('saved jpg')
# fig.savefig(f'fit_multi-experiment005--Ensemble-3D-Var-figures/fe005--Ensemble-3D-Var-compare_analyses-{name_addon}.pdf')
# print('saved pdf')