#!/usr/bin/env python

# Find a point in latent space that maximises the fit to a set of pseudo-obs,
#  and plot the fitted state.
# Make multiple fits and plot the ensemble.

import os
import sys
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

import random

import iris
import IRData.twcr as twcr
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
parser.add_argument("--ensemble", help="Ensemble size", type=int, required=False, default=150)
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


date = datetime.datetime(year=args.year, month=args.month, day=args.day)

t = ERA5_load_T850(args.year, args.month, args.day)
t_orig = t.copy()
c = ERA5_load_T850_climatology(args.year, args.month, args.day)
vc = ERA5_load_T850_variability_climatology(args.year, args.month, args.day)
t = t - c
t = t / vc
t = ERA5_roll_longitude(t)
# t = ERA5_trim(t)
t_in = tf.convert_to_tensor(t.data, np.float32)
t_in = tf.reshape(t_in, [1, 720, 1440, 1])

vc = ERA5_roll_longitude(vc)
vc = vc.data

# Set up the model and load the weights at the chosen epoch
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

means, successes= [], []
epochs = []
constant = 1.0



autoencoder = DCVAE()

weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
    args.epoch,
)
print('Uploading weights from', weights_dir)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
# Check the load worked
devn = load_status.assert_existing_objects_matched()
print('Weights loaded, objects match')

# We are using it in inference mode
# (I'm not at all sure this actually works)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()
print('Autoencoder compiled')

B_matrix = pickle.load(
    open('../validation/decoding_experiment006---B-matrix_for_persistence-data/B_matrix_2015-01-01_to_2018-12-31.pkl',
         'rb'))

#latent = tf.Variable(constant * np.ones(shape=(1, autoencoder.latent_dim)))    # Bostjan: For constant values of all members of latent space
#fitted = autoencoder.sample_call(t_in, size=args.ensemble)  # TO SE NI OK
latent_mean_orig, latent_logvar_orig = autoencoder.encode(t_in)

latent_samples = tf.Variable(tf.random.normal(mean=latent_mean_orig, stddev=np.sqrt(np.diagonal(B_matrix)), shape=(args.ensemble, autoencoder.latent_dim)))
decoded = autoencoder.decode(latent_samples)
print('Decoded')

e_mean = tf.math.reduce_mean(decoded, axis=0)
e_std = tf.math.reduce_std(decoded, axis=0)
#e_std = (e_std.numpy() - 0) * 15


fig = Figure(
    figsize=(5, 8),
    dpi=300,
    facecolor=(0.88, 0.88, 0.88, 1),
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
cbars = 1
tot_plots = 2
height_sum = tot_plots * 0.075 + cbars * 0.015 + 0.005 * (tot_plots + cbars + 1)
dheight_plot = 0.075 / height_sum
dheight_cbar = 0.015 / height_sum
dheight_buffer = 0.005 / height_sum
plot_bottom = 1


lm = get_land_mask()
# 1st (=top) - original field, obs. points
plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
ax_std = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
ax_std.set_aspect("equal")
ax_std.set_xticks([])
ax_std.set_yticks([])
ax_std.set_xlim(-180, 180)
ax_std.set_ylim(-90, 90)
ax_std.set_ylabel('%04d-%02d-%02d' % (args.year, args.month, args.day))
stdp = plot_Earth(
    ax_std,
    tf.squeeze(e_std).numpy() * vc,
    vMin=0,
    vMax=5,
    land=lm,
    #label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
    cmap='terrain_r'
)

plot_bottom += dheight_plot + dheight_buffer
ax_stdcb = fig.add_axes([(1-0.95)/2, plot_bottom, 0.95, dheight_cbar])
plot_colourbar(fig, ax_stdcb, stdp, ticks=[0,1,2,3,4,5])
plot_bottom -= dheight_plot + dheight_buffer

lm = get_land_mask()
fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_std).numpy() * vc, transform=ccrs.PlateCarree(),
                    vmin=0, vmax=5,
                    cmap='terrain_r')
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='max', label=r'$\sigma(T_{850})$ [$\degree$C]')
ax2.set_title('Backg. std %04d-%02d-%02d' % (args.year, args.month, args.day), y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    'decoding_experiment008--Background_uncertainty-figures/de008--Background_uncertainty-%04d-%02d-%02d'
    % (args.year, args.month, args.day) + '.jpg', dpi=300)

#2nd variability climatology
plot_bottom -= dheight_plot + dheight_buffer
ax_vc = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
ax_vc.set_aspect("equal")
ax_vc.set_xticks([])
ax_vc.set_yticks([])
ax_vc.set_xlim(-180, 180)
ax_vc.set_ylim(-90, 90)
ax_vc.set_ylabel('Climatological std %02d-%02d' % (args.month, args.day))
vcp = plot_Earth(
    ax_vc,
    vc,
    vMin=0,
    vMax=5,
    land=lm,
    #label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
    cmap='terrain_r'
)
fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, vc, transform=ccrs.PlateCarree(),
                    vmin=0, vmax=5,
                    cmap='terrain_r')
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='max', label=r'$\sigma(T_{850})$ [$\degree$C]')
ax2.set_title('Clim. std %02d-%02d' % (args.month, args.day), y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(
    'decoding_experiment008--Background_uncertainty-figures/de008--Background_uncertainty-%04d-%02d-%02d-vc'
    % (args.year, args.month, args.day) + '.jpg', dpi=300)


fig.savefig(f"decoding_experiment008--Background_uncertainty-figures/de008--Background_uncertainty-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}.jpg", facecolor="white")
finish = datetime.datetime.now()
print(str(finish - start))
print('saved jpg')
# fig.savefig(f"decoding_experiment008--Background_uncertainty-figures/de008--Background_uncertainty-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}.pdf", facecolor="white")
# finish = datetime.datetime.now()
# print(str(finish - start))
# print('saved pdf')
