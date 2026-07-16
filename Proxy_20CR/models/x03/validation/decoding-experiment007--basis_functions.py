#!/usr/bin/env python

# Find a point in latent space that maximises the fit to a set of pseudo-obs,
#  and plot the fitted state.
# Make multiple fits and plot the ensemble.

import os
import sys
import numpy as np
import scipy.stats

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


plot_gridpoint = True
plot_histogram = False
compute_histogram = False
ppf = False
date_for_title = True#False
perturb = 0.1
vmin = -0.2
vmax = 0.2

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

#latent = tf.Variable(constant * np.ones(shape=(1, autoencoder.latent_dim)))    # Bostjan: For constant values of all members of latent space
#fitted = autoencoder.sample_call(t_in, size=args.ensemble)  # TO SE NI OK
latent_mean_orig, latent_logvar_orig = autoencoder.encode(t_in)
print('Encoded')
latent_sigma_orig = tf.exp(latent_logvar_orig * 0.5)
latent_samples = tf.Variable(tf.random.normal(mean=latent_mean_orig, stddev=0.0, shape=(1, autoencoder.latent_dim)))
print('Sampled')
decoded_renormalized = tf.squeeze(autoencoder.decode(latent_samples)) * vc
print('Decoded')

lm = get_land_mask()
transform = ccrs.Robinson()
fig2 = plt.figure(figsize=(6, 4))
matplotlib.rcParams.update({"font.size": 16})
ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
lats = lm.coord("latitude").points
lons = lm.coord("longitude").points
lons, lats = np.meshgrid(lons, lats)
print('entering pc')
pc = ax2.pcolormesh(lons, lats, decoded_renormalized, transform=ccrs.PlateCarree(), vmin=-10, vmax=10, cmap='seismic')
print('adding coastlines')
ax2.coastlines()
print('setting global')
ax2.set_global()
fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
if not date_for_title:
    ax2.set_title(r"$D(\mathbf{z}=\mu_\phi(\mathbf{x}^t))$", y=1.02)#('Decoded mean', y=1.02)
else:
    ax2.set_title(r"$D(\mathbf{z}=\mu_\phi(\mathbf{x}^t))$" + ' for %04d-%02d-%02d' % (args.year, args.month, args.day), y=1.02)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
)
print('saving')
fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_original.jpg', dpi=300
)
# fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_original.pdf')
print('saved single fig.')
raise AssertionError



if plot_gridpoint:
    for idx in range(autoencoder.latent_dim):
    #idx = 1
        if idx%15 == 0:
            fig2 = plt.figure(figsize=(18,20))
        print('Chosen idx', idx)
        zeros = np.zeros((1,100))
        inc_plus1 = zeros.copy()
        inc_plus1[0][idx] = inc_plus1[0][idx] + perturb
        #print(inc_plus1)
        latent_plus1 = latent_mean_orig + inc_plus1
        #print(latent_plus1)
        latent_samples_plus1 = tf.Variable(tf.random.normal(mean=latent_plus1, stddev=0.0, shape=(1, autoencoder.latent_dim)))
        decoded_plus1_renormalized = tf.squeeze(autoencoder.decode(latent_samples_plus1)) * vc

        difference_plus_1 = decoded_plus1_renormalized - decoded_renormalized
        #print(difference_plus_1[0,204,411])

        # zeros = np.zeros((1,100))
        # inc_minus1 = zeros.copy()
        # inc_minus1[0][idx] = inc_minus1[0][idx] - 1
        # #print(inc_minus1)
        # latent_minus1 = latent_mean_orig + inc_minus1
        # #print(latent_minus1)
        # latent_samples_minus1 = tf.Variable(tf.random.normal(mean=latent_minus1, stddev=0.0, shape=(1, autoencoder.latent_dim)))
        # decoded_minus1_renormalized = tf.squeeze(autoencoder.decode(latent_samples_minus1)) * vc
        #
        # difference_minus_1 = decoded_minus1_renormalized - decoded_renormalized
        # print(np.amax(difference_plus_1 + difference_minus_1), np.amin(difference_plus_1 + difference_minus_1))
        #
        # zeros = np.zeros((1,100))
        # inc_plus02 = zeros.copy()
        # inc_plus02[0][0] = inc_plus02[0][0] + 0.2
        # latent_plus02 = latent_mean_orig + inc_plus02
        # latent_samples_plus02 = tf.Variable(tf.random.normal(mean=latent_plus02, stddev=0.0, shape=(1, autoencoder.latent_dim)))
        # decoded_plus02_renormalized = tf.squeeze(autoencoder.decode(latent_samples_plus02)) * vc
        #
        # difference_plus_02 = decoded_plus1_renormalized - decoded_renormalized
        #
        # zeros = np.zeros((1,100))
        # inc_minus02 = zeros.copy()
        # inc_minus02[0][0] = inc_minus02[0][0] - 0.2
        # latent_minus02 = latent_mean_orig + inc_minus02
        # latent_samples_minus02 = tf.Variable(tf.random.normal(mean=latent_minus02, stddev=0.0, shape=(1, autoencoder.latent_dim)))
        # decoded_minus02_renormalized = tf.squeeze(autoencoder.decode(latent_samples_minus02)) * vc
        #
        # difference_minus_02 = decoded_minus02_renormalized - decoded_renormalized

        #
        # fig = Figure(
        #     figsize=(15, 20),
        #     dpi=300,
        #     facecolor="white",
        #     edgecolor=None,
        #     linewidth=0.0,
        #     frameon=False,
        #     subplotpars=None,
        #     tight_layout=None,
        # )
        # canvas = FigureCanvas(fig)
        # matplotlib.rcParams.update({"font.size": 16})
        #
        # ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
        # ax_global.set_axis_off()
        # ax_global.autoscale(enable=False)
        # ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")
        #
        # plot_left = 0.075
        # plot_width = 0.85
        # cbars = 1
        # tot_plots = 5
        # height_sum = tot_plots * 0.075 + cbars * 0.015 + 0.005 * (tot_plots + cbars)
        # dheight_plot = 0.075 / height_sum
        # dheight_cbar = 0.015 / height_sum
        # dheight_buffer = 0.005 / height_sum
        # plot_bottom = 1
        #
        # lm = get_land_mask()
        # # 1st (=top) - original field
        # plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
        # ax_of = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
        # ax_of.set_aspect("equal")
        # ax_of.set_xticks([])
        # ax_of.set_yticks([])
        # ax_of.set_xlim(-180, 180)
        # ax_of.set_ylim(-90, 90)
        # ax_of.set_ylabel('Decoded mean for %04d-%02d-%02d' % (args.year, args.month, args.day) + r' [$\degree$C]')
        # ofp = plot_Earth(
        #     ax_of,
        #     decoded_renormalized,
        #     vMin=-10,
        #     vMax=10,
        #     land=lm,
        #     # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
        # )
        #
        # plot_bottom += dheight_plot + dheight_buffer
        # ax_ocb = fig.add_axes([(1 - 0.5) / 2, plot_bottom, 0.5, dheight_cbar])
        # plot_colourbar(fig, ax_ocb, ofp)
        # plot_bottom -= dheight_plot - dheight_buffer
        #
        # # 2nd
        # plot_bottom -= dheight_plot + dheight_buffer
        # ax_plus1 = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
        # ax_plus1.set_aspect("equal")
        # ax_plus1.set_xticks([])
        # ax_plus1.set_yticks([])
        # ax_plus1.set_xlim(-180, 180)
        # ax_plus1.set_ylim(-90, 90)
        # ax_plus1.set_ylabel('Increment for larger z[0] + 1')
        # plus1_p = plot_Earth(
        #     ax_plus1,
        #     decoded_plus1_renormalized - decoded_renormalized,
        #     vMin=-10,
        #     vMax=10,
        #     land=lm,
        #     # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
        # )
        #
        # # 3rd
        # plot_bottom -= dheight_plot + dheight_buffer
        # ax_plus02 = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
        # ax_plus02.set_aspect("equal")
        # ax_plus02.set_xticks([])
        # ax_plus02.set_yticks([])
        # ax_plus02.set_xlim(-180, 180)
        # ax_plus02.set_ylim(-90, 90)
        # ax_plus02.set_ylabel('Increment for larger z[0] + 0.2')
        # plus02_p = plot_Earth(
        #     ax_plus02,
        #     decoded_plus02_renormalized - decoded_renormalized,
        #     vMin=-10,
        #     vMax=10,
        #     land=lm,
        #     # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
        # )
        #
        # # 4th
        # plot_bottom -= dheight_plot + dheight_buffer
        # ax_minus02 = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
        # ax_minus02.set_aspect("equal")
        # ax_minus02.set_xticks([])
        # ax_minus02.set_yticks([])
        # ax_minus02.set_xlim(-180, 180)
        # ax_minus02.set_ylim(-90, 90)
        # ax_minus02.set_ylabel('Increment for larger z[0] - 0.2')
        # minus02_p = plot_Earth(
        #     ax_minus02,
        #     decoded_minus02_renormalized - decoded_renormalized,
        #     vMin=-10,
        #     vMax=10,
        #     land=lm,
        #     # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
        # )
        #
        # # 5th
        # plot_bottom -= dheight_plot + dheight_buffer
        # ax_minus1 = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
        # ax_minus1.set_aspect("equal")
        # ax_minus1.set_xticks([])
        # ax_minus1.set_yticks([])
        # ax_minus1.set_xlim(-180, 180)
        # ax_minus1.set_ylim(-90, 90)
        # ax_minus1.set_ylabel('Increment for z[0] - 1')
        # minus1_p = plot_Earth(
        #     ax_minus1,
        #     decoded_minus1_renormalized - decoded_renormalized,
        #     vMin=-10,
        #     vMax=10,
        #     land=lm,
        #     # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
        # )
        #
        # savefig_name = f'decoding_experiment007--Basis_functions-figures/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}'
        # fig.savefig(savefig_name + ".jpg", dpi=300, facecolor="white")
        # print('saved jpg')
        #
        #
        #
        # fig = Figure(
        #     figsize=(15, 4),
        #     dpi=300,
        #     facecolor="white",
        #     edgecolor=None,
        #     linewidth=0.0,
        #     frameon=False,
        #     subplotpars=None,
        #     tight_layout=None,
        # )
        # canvas = FigureCanvas(fig)
        # matplotlib.rcParams.update({"font.size": 16})
        #
        # ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
        # ax_global.set_axis_off()
        # ax_global.autoscale(enable=False)
        # ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")
        #
        # plot_left =0
        # plot_width = 0.85
        # cbars = 1
        # tot_plots = 3
        # width_sum = tot_plots * 0.85 + 0.10 * (tot_plots + 2)
        # dleft_plot = 0.85 / width_sum
        # dleft_buffer = 0.10 / width_sum
        # plot_left = 0
        # plot_bottom = 0.05
        # cbar_bottom = 0.83
        # height_plot = 0.75
        # height_cbar = 0.15
        # width_cbar = dleft_plot * 0.8
        #
        lm = get_land_mask()
        # # 1st (=top) - original field
        # plot_left += dleft_buffer# dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
        # ax_of = fig.add_axes([plot_left, plot_bottom, dleft_plot, height_plot])
        # ax_of.set_aspect("equal")
        # ax_of.set_xticks([])
        # ax_of.set_yticks([])
        # ax_of.set_xlim(-180, 180)
        # ax_of.set_ylim(-90, 90)
        # ax_of.set_ylabel('Decoded mean') # for %04d-%02d-%02d' % (args.year, args.month, args.day) + r' [$\degree$C]'
        # ofp = plot_Earth(
        #     ax_of,
        #     decoded_renormalized,
        #     vMin=-10,
        #     vMax=10,
        #     land=lm,
        #     # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
        # )
        #
        # ax_ocb = fig.add_axes([plot_left + (dleft_plot - width_cbar)/2, cbar_bottom, width_cbar, height_cbar])
        # plot_colourbar(fig, ax_ocb, ofp)
        #
        # transform = ccrs.Robinson()
        # fig2 = plt.figure(figsize=(6, 4))
        # ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
        # lats = lm.coord("latitude").points
        # lons = lm.coord("longitude").points
        # lons, lats = np.meshgrid(lons, lats)
        # print('entering pc')
        # pc = ax2.pcolormesh(lons, lats, decoded_renormalized, transform=ccrs.PlateCarree(), vmin=-10, vmax=10, cmap='seismic')
        # print('adding coastlines')
        # ax2.coastlines()
        # print('setting global')
        # ax2.set_global()
        # fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
        # if not date_for_title:
        #     ax2.set_title('Decoded mean', y=1.02)
        # else:
        #     ax2.set_title('Decoded mean for %04d-%02d-%02d' % (args.year, args.month, args.day), y=1.02)
        # gl = ax2.gridlines(
        #     draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        # )
        # print('saving')
        # fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_original.jpg', dpi=300
        # )
        # # fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_original.pdf')
        # print('saved single fig.')
        #
        # # 2nd
        # plot_left += dleft_plot + dleft_buffer
        # ax_plus1 = fig.add_axes([plot_left, plot_bottom, dleft_plot, height_plot])
        # ax_plus1.set_aspect("equal")
        # ax_plus1.set_xticks([])
        # ax_plus1.set_yticks([])
        # ax_plus1.set_xlim(-180, 180)
        # ax_plus1.set_ylim(-90, 90)
        # ax_plus1.set_ylabel(f'Diff. if larger z[{idx}]')
        # plus1_p = plot_Earth(
        #     ax_plus1,
        #     decoded_plus1_renormalized - decoded_renormalized,
        #     vMin=-2,
        #     vMax=2,
        #     land=lm,
        #     # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
        # )
        # axcbplus1 = fig.add_axes([plot_left + (dleft_plot - width_cbar)/2, cbar_bottom, width_cbar, height_cbar])
        # plot_colourbar(fig, axcbplus1, plus1_p)


        transform = ccrs.Robinson()
        # fig2 = plt.figure(figsize=(6, 4))
        #ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
        ax2 = fig2.add_subplot(5, 3, idx%15+1, projection=transform)
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        print('entering pc')
        pc = ax2.pcolormesh(lons, lats, decoded_plus1_renormalized - decoded_renormalized, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='seismic')
        print('adding coastlines')
        ax2.coastlines()
        print('setting global')
        ax2.set_global()
        fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
        if not date_for_title:
            ax2.set_title(r'Diff. if $\mathbf{z}$' + f'[{idx}]' + r'$\mapsto$ $\mathbf{z}$' + f'[{idx}] + {perturb}', y=1.02)
        else:
            ax2.set_title('Diff. for %04d-%02d-%02d' % (args.year, args.month, args.day), y=1.02)
            ax2.set_title(r'Diff. if $\mathbf{z}$' + f'[{idx}]' + r'$\mapsto$ $\mathbf{z}$' + f'[{idx}] + {perturb}', y=1.02)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        print('saving')
        # fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{idx}+1.jpg', dpi=300
        # )
        if idx%15 == 14:
            fig2.savefig(f'decoding_experiment007--Basis_functions-figures/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{idx-14}_to_{idx}+{perturb}.jpg', dpi=300)
        # fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_+1.pdf'
        # )
        print('saved single fig.')
    raise AssertionError # Don't need other stuff right now

    # 3rd
    plot_left += dleft_plot + dleft_buffer
    ax_minus1 = fig.add_axes([plot_left, plot_bottom, dleft_plot, height_plot])
    ax_minus1.set_aspect("equal")
    ax_minus1.set_xticks([])
    ax_minus1.set_yticks([])
    ax_minus1.set_xlim(-180, 180)
    ax_minus1.set_ylim(-90, 90)
    ax_minus1.set_ylabel(f'Diff. if smaller z[{idx}]')
    minus1_p = plot_Earth(
        ax_minus1,
        decoded_minus1_renormalized - decoded_renormalized,
        vMin=-2,
        vMax=2,
        land=lm,
        # label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
    )
    axcbminus1 = fig.add_axes([plot_left + (dleft_plot - width_cbar)/2, cbar_bottom, width_cbar, height_cbar])
    plot_colourbar(fig, axcbminus1, minus1_p)

    transform = ccrs.Robinson()
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
    lats = lm.coord("latitude").points
    lons = lm.coord("longitude").points
    lons, lats = np.meshgrid(lons, lats)
    print('entering pc')
    pc = ax2.pcolormesh(lons, lats, decoded_minus1_renormalized - decoded_renormalized, transform=ccrs.PlateCarree(), vmin=-2, vmax=2, cmap='seismic')
    print('adding coastlines')
    ax2.coastlines()
    print('setting global')
    ax2.set_global()
    fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both', label=r'$\Delta T_{850}$ [$\degree$C]')
    ax2.set_title(r'Diff. if $\mathbf{z}$' + f'[{idx}]' + r'$\mapsto$ $\mathbf{z}$' + f'[{idx}] - 1', y=1.02)
    gl = ax2.gridlines(
        draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    print('saving')
    fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{idx}-1.jpg', dpi=300
    )
    # fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_-1.pdf'
    # )
    print('saved single fig.')


    # 4th Mean difference
    transform = ccrs.Robinson()
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
    lats = lm.coord("latitude").points
    lons = lm.coord("longitude").points
    lons, lats = np.meshgrid(lons, lats)
    print('entering pc')
    pc = ax2.pcolormesh(lons, lats, (difference_plus_1 + difference_minus_1)/2, transform=ccrs.PlateCarree(),
                        vmin=-0.2, vmax=0.2, cmap='seismic')
    print('adding coastlines')
    ax2.coastlines()
    print('setting global')
    ax2.set_global()
    fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                  label=r'$\Delta T_{850}$ [$\degree$C]')
    ax2.set_title(r'Mean diff.', y=1.02)
    gl = ax2.gridlines(
        draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    print('saving')
    fig2.savefig(
        f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{idx}m+-1.jpg',
        dpi=300
        )
    # fig2.savefig(f'decoding_experiment007--Basis_functions-figures/single_figs/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_-1.pdf'
    # )
    print('saved single fig.')


    savefig_name = f'decoding_experiment007--Basis_functions-figures/de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_horizontal'
    #fig.savefig(savefig_name + ".jpg", dpi=300, facecolor="white")
    print('saved jpg')



plt.cla()
plt.clf()
if plot_histogram:
    N = 100 # * 150
    M = 300
    idx = 0
    my_loc = [204, 411] # [38.875N, 77.125E]
    name = f'de007--Basis_functions-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_perturbed_{idx}_row={my_loc[0]}_col={my_loc[1]}'
    if ppf:
        name += f'_NM={N*M}_ppf'
    else:
        name += f'_NM={N * M}'
    if compute_histogram:
        all_at_point = []
        if not ppf:
            for i in range(N):
                print(i)
                zeros = np.zeros((M, 100))
                inc = zeros.copy()
                inc[:,idx] = inc[:,idx] + np.random.normal(loc=0, scale=1, size=(M))
                #print('inc', inc)
                latent = latent_mean_orig + inc
                #print('latent', latent)
                decoded_renormalized_samples = tf.squeeze(autoencoder.decode(latent)) * vc
                all_at_point.append(decoded_renormalized_samples[:, my_loc[0], my_loc[1]] - decoded_renormalized[my_loc[0], my_loc[1]])
        else:
            for i in range(N):
                print(i)
                zeros = np.zeros((M, 100))
                inc = zeros.copy()
                try:
                    inc[:, idx] = inc[:, idx] + scipy.stats.norm.ppf(np.linspace(1/(N*M),1-1/(N*M),N*M-1)[M*i:M*(i+1)])
                except:
                    inc[:, idx] = inc[:, idx] + scipy.stats.norm.ppf(
                        np.linspace(1 / (N * M), 1 - 1 / (N * M), N * M - 1)[M * i:-1])
                # print('inc', inc)
                latent = latent_mean_orig + inc
                # print('latent', latent)
                decoded_renormalized_samples = tf.squeeze(autoencoder.decode(latent)) * vc
                all_at_point.append(
                    decoded_renormalized_samples[:, my_loc[0], my_loc[1]] - decoded_renormalized[my_loc[0], my_loc[1]])
        all_at_point = np.array(all_at_point).flatten()
        pickle.dump(all_at_point, open('decoding_experiment007--Basis_functions-data/' + name + '.pkl', 'wb'))
    else:
        all_at_point = pickle.load(open('decoding_experiment007--Basis_functions-data/' + name + '.pkl', 'rb'))

    #print(all_at_point)
    print(np.shape(all_at_point))
    plt.hist(all_at_point, bins=np.linspace(-2.1, 2.1, 85))
    plt.grid()
    plt.savefig('decoding_experiment007--Basis_functions-figures/' + name + '.pdf')
