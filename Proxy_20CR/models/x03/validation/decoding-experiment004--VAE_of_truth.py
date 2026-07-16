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
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()


latent_mean_orig, latent_logvar_orig = autoencoder.encode(t_in)
latent_sigma_orig = tf.exp(latent_logvar_orig * 0.5)
latent_samples = tf.Variable(tf.random.normal(mean=latent_mean_orig, stddev=latent_sigma_orig, shape=(args.ensemble, autoencoder.latent_dim)))
decoded = autoencoder.decode(latent_samples)

e_mean = tf.math.reduce_mean(decoded, axis=0)
e_std = tf.math.reduce_std(decoded, axis=0)
e_mean = tf.squeeze(e_mean).numpy() * vc
print('got e_mean')
#e_std = (e_std.numpy() - 0) * 15


fig = Figure(
    figsize=(15, 25),
    dpi=100,
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

lm = get_land_mask()
# 1st (=top) - original field, obs. points
ax_of = fig.add_axes([0.075, 0.82, 0.85, 0.17])
ax_of.set_aspect("equal")
ax_of.set_xticks([])
ax_of.set_yticks([])
ax_of.set_xlim(-180, 180)
ax_of.set_ylim(-90, 90)
ax_of.set_ylabel('Truth for '+ date.strftime('%d %b %Y') + r' [$\degree$C]')
ofp = plot_Earth(
    ax_of,
    tf.squeeze(t_in).numpy() * vc,
    vMin=-10,
    vMax=10,
    land=lm,
    #label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
)

ax_ocb = fig.add_axes([0.115, 0.79, 0.77, 0.02])
plot_colourbar(fig, ax_ocb, ofp)
print('got 1st')

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
pc = ax2.pcolormesh(lons, lats, tf.squeeze(t_in).numpy() * vc, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                    cmap='seismic')
ax2.coastlines()
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
ax2.set_title(f'Truth for %04d-%02d-%02d' % (args.year, args.month, args.day), y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
fig2.savefig(
    "decoding_experiment004--VAE_of_truth-figures/single_figs/de004--VAE_of_truth-%04d-%02d-%02d_epoch=%04d_ensemble=%d-truth.jpg"
    % (args.year, args.month, args.day, args.epoch, args.ensemble), dpi=300)

# 2nd - mean output field
ax_mof = fig.add_axes([0.075, 0.60, 0.85, 0.17])
ax_mof.set_aspect("equal")
ax_mof.set_xticks([])
ax_mof.set_yticks([])
ax_mof.set_xlim(-180, 180)
ax_mof.set_ylim(-90, 90)
ax_mof.set_ylabel(r"Mean output field [$\degree$C]")
efp = plot_Earth(
    ax_mof,
    e_mean,
    vMin=-10,
    vMax=10,
    #fog=tf.squeeze((e_std / c_std)).numpy(),
    #fog_threshold=0.33,
    land=lm,
    #label="Difference: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
)

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
pc = ax2.pcolormesh(lons, lats, e_mean, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                    cmap='seismic')
ax2.coastlines()
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
ax2.set_title(f'Mean VAE(truth)', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
fig2.savefig(
    "decoding_experiment004--VAE_of_truth-figures/single_figs/de004--VAE_of_truth-%04d-%02d-%02d_epoch=%04d_ensemble=%d-mean_VAE_of_turth.jpg"
    % (args.year, args.month, args.day, args.epoch, args.ensemble), dpi=300)

# 3rd - mean output field - original field
ax_ef = fig.add_axes([0.075, 0.42, 0.85, 0.17])
ax_ef.set_aspect("equal")
ax_ef.set_xticks([])
ax_ef.set_yticks([])
ax_ef.set_xlim(-180, 180)
ax_ef.set_ylim(-90, 90)
ax_ef.set_ylabel(r"Mean output field - truth [$\degree$C]")
efp = plot_Earth(
    ax_ef,
    e_mean - tf.squeeze(t_in).numpy() * vc,
    vMin=-10,
    vMax=10,
    #fog=tf.squeeze((e_std / c_std)).numpy(),
    #fog_threshold=0.33,
    land=lm,
    #label="Difference: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
)

print('got 3rd')

# 4th box chart of latent variables
ax_box = fig.add_axes([0.075, 0.24, 0.85, 0.17])
# narisi razliko med enim od fitted in pa povprecjem od fitted
one_gp_sample = tf.squeeze(decoded[0]).numpy() * vc
ax_box.set_aspect("equal")
ax_box.set_xticks([])
ax_box.set_yticks([])
ax_box.set_xlim(-180, 180)
ax_box.set_ylim(-90, 90)
ax_box.set_ylabel(r"Single out. field - mean out. field [$\degree$C]")
efp = plot_Earth(
    ax_box,
    one_gp_sample - e_mean,
    vMin=-10,
    vMax=10,
    #fog=tf.squeeze((e_std / c_std)).numpy(),
    #fog_threshold=0.33,
    land=lm,
    #label="Difference: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
)

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
pc = ax2.pcolormesh(lons, lats, one_gp_sample - e_mean, transform=ccrs.PlateCarree(), vmin=-1, vmax=1,
                    cmap='seismic')
ax2.coastlines()
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
ax2.set_title(f'One realisation - mean VAE(truth)', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
fig2.savefig(
    "decoding_experiment004--VAE_of_truth-figures/single_figs/de004--VAE_of_truth-%04d-%02d-%02d_epoch=%04d_ensemble=%d-single_minus_mean_VAE_of_turth.jpg"
    % (args.year, args.month, args.day, args.epoch, args.ensemble), dpi=300)

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
pc = ax2.pcolormesh(lons, lats, one_gp_sample, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                    cmap='seismic')
ax2.coastlines()
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
ax2.set_title(f'One realisation of VAE(truth)', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
fig2.savefig(
    "decoding_experiment004--VAE_of_truth-figures/single_figs/de004--VAE_of_truth-%04d-%02d-%02d_epoch=%04d_ensemble=%d-single_VAE_of_turth.jpg"
    % (args.year, args.month, args.day, args.epoch, args.ensemble), dpi=300)


print('got 4th')

# 5th std of output field
# c_std = ERA5_load_T850_variability_climatology(1981, args.month, args.day)
# c_std /= 15
# c_std = ERA5_roll_longitude(c_std)
# c_std = ERA5_trim(c_std)
# c_std = tf.convert_to_tensor(c_std.data, np.float32)
# c_std = tf.reshape(c_std, [1, 720, 1440, 1])
ax_std = fig.add_axes([0.075, 0.02, 0.85, 0.17])
ax_std.set_aspect("equal")
ax_std.set_xticks([])
ax_std.set_yticks([])
ax_std.set_xlim(-180, 180)
ax_std.set_ylim(-90, 90)
ax_std.set_ylabel(r'Standard deviation [$\degree$C]')
stdp = plot_Earth(
    ax_std,
    tf.squeeze(e_std).numpy() * vc,
    vMin=0,
    vMax=0.6,#np.amax(tf.squeeze(e_std).numpy() * vc),
    fog=None,#tf.squeeze((e_std / c_std)).numpy(),
    fog_threshold=0.1,
    land=lm,
    #label="Uncertainty: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
    cmap='terrain_r'
)
ax_stdcb = fig.add_axes([0.115, 0.21, 0.77, 0.02])
plot_colourbar(fig, ax_stdcb, stdp)
# pickle.dump((t_in, e_mean, c_std, e_std, t_obs), open("tst.pkl", "wb"))
print('got 5th')

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_std).numpy() * vc, transform=ccrs.PlateCarree(), vmin=0, vmax=0.5,
                    cmap='terrain_r')
ax2.coastlines()
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='max', label=r'$\sigma(T_{850})$ [$\degree$C]')
ax2.set_title(f'Std of VAE(truth)', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
fig2.savefig(
    "decoding_experiment004--VAE_of_truth-figures/single_figs/de004--VAE_of_truth-%04d-%02d-%02d_epoch=%04d_ensemble=%d-std.jpg"
    % (args.year, args.month, args.day, args.epoch, args.ensemble), dpi=300)



fig.savefig(f"decoding_experiment004--VAE_of_truth-figures/de004--VAE_of_truth-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}.jpg", dpi=300, facecolor="white")
finish = datetime.datetime.now()
print(str(finish - start))
print('saved jpg')
#fig.savefig(f"decoding_experiment004-figures/decoding_experiment004_{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}.pdf", facecolor="white")
finish = datetime.datetime.now()
print(str(finish - start))
#print('saved pdf')


fig, ax_box = plt.subplots(figsize=(10,6))
bpltrue = ax_box.boxplot(latent_samples.numpy(), whis=[5,95], showfliers=False)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bpltrue[element], color='C1')

ax_box.set_xticks([i for i in range(1, autoencoder.latent_dim+1) if (i-1)%5==0], [i for i in range(autoencoder.latent_dim) if i%5==0])
ax_box.grid(linestyle=':', linewidth=0.6, color='gray')
ax_box.set_ylabel('Latent element value')
ax_box.set_xlabel('Index in latent space')
ax_box.legend([bpltrue["boxes"][0]], ['Encoded truth'], loc='upper right', framealpha=0.6)
fig.savefig(f"decoding_experiment004--VAE_of_truth-figures/de004--VAE_of_truth-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}_latent.jpg")
finish = datetime.datetime.now()
print(str(finish - start))
print('saved jpg')
fig.savefig(f"decoding_experiment004--VAE_of_truth-figures/de004--VAE_of_truth-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}_latent.pdf", facecolor="white")
finish = datetime.datetime.now()
print(str(finish - start))
print('saved pdf')



latent_samples_mean = tf.Variable(tf.random.normal(mean=latent_mean_orig, stddev=0.0, shape=(1, autoencoder.latent_dim)))
decoded_mean = autoencoder.decode(latent_samples_mean)
e_from_mean = tf.math.reduce_mean(decoded_mean, axis=0)
e_from_mean = tf.squeeze(e_from_mean).numpy() * vc

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
pc = ax2.pcolormesh(lons, lats, e_from_mean - e_mean, transform=ccrs.PlateCarree(), vmin=-1, vmax=1,
                    cmap='seismic')
ax2.coastlines()
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
ax2.set_title(r'Decoded $\mu_\phi(\mathbf{x}^\mathrm{true})$ - mean VAE(truth)', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
fig2.savefig(
    "decoding_experiment004--VAE_of_truth-figures/single_figs/de004--VAE_of_truth-%04d-%02d-%02d_epoch=%04d_ensemble=%d-decoded_mean_minus_mean_VAE_of_turth.jpg"
    % (args.year, args.month, args.day, args.epoch, args.ensemble), dpi=300)

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
pc = ax2.pcolormesh(lons, lats, e_from_mean, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                    cmap='seismic')
ax2.coastlines()
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
ax2.set_title(r'Decoded $\mu_\phi(\mathbf{x}^\mathrm{true})$', y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
fig2.savefig(
    "decoding_experiment004--VAE_of_truth-figures/single_figs/de004--VAE_of_truth-%04d-%02d-%02d_epoch=%04d_ensemble=%d-decoded_mean.jpg"
    % (args.year, args.month, args.day, args.epoch, args.ensemble), dpi=300)