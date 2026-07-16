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
parser.add_argument("--year", help="Year", type=int, required=False, default=2019)
parser.add_argument("--month", help="Integer month", type=int, required=False, default=4)
parser.add_argument("--day", help="Day of month", type=int, required=False, default=15)
parser.add_argument("--center_option",
                    help="Which vector is subtracted from each ensemble member before computing the outer product or the B-matrix; 'ensemble_latent_mean' for mean of the 9 ensemble members; 'control' for the control member;",
                    type=str, required=True)
parser.add_argument("--plot", help='Plot B matrix and histogram of its elements', default=False, action=argparse.BooleanOptionalAction)

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

# Set up the model and load the weights at the chosen epoch
print('PREPARING VAE!')
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % (1000*(args.epoch//1000))]
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

c = ERA5_load_T850_climatology(args.year, args.month, args.day)
vc = ERA5_load_T850_variability_climatology(args.year, args.month, args.day)
vc = ERA5_roll_longitude(vc)
vc = vc.data

t_ens_members = []
for iens_member in range(1, 9+1):
    print('ens member', iens_member)
    t_iens_member = ERA5_load_T850(args.year, args.month, args.day, ensemble_member=iens_member)
    t_iens_member = t_iens_member - c
    t_iens_member = ERA5_roll_longitude(t_iens_member)
    t_iens_member = t_iens_member.data
    t_iens_member = t_iens_member / vc
    t_ens_members.append(t_iens_member)
t_ens_members_in = tf.convert_to_tensor(t_ens_members, np.float32)
t_ens_members_in = tf.reshape(t_ens_members_in, [9, 720, 1440, 1])
latent_ens_members_mean, latent_ens_members_logvar = autoencoder.encode(t_ens_members_in)
latent_ens_members_npy = latent_ens_members_mean.numpy()

if args.center_option == 'control':
    t_control = ERA5_load_T850(args.year, args.month, args.day, ensemble_member='control')
    t_control = t_control - c
    t_control = ERA5_roll_longitude(t_control)
    t_control_origish = t_control.copy()
    t_control = t_control / vc
    t_control_in = tf.convert_to_tensor(t_control.data, np.float32)
    t_control_in = tf.reshape(t_control_in, [1, 720, 1440, 1])
    latent_control_mean, latent_control_logvar = autoencoder.encode(t_control_in)
    latent_center_npy = latent_control_mean.numpy()
    # print('np.shape(latent_center_npy)', np.shape(latent_center_npy))

if args.center_option == 'ensemble_latent_mean':
    latent_center_npy = np.expand_dims(np.mean(latent_ens_members_npy, axis=0), axis=0)
    # print('np.shape(latent_center_npy)', np.shape(latent_center_npy))

matrices = []
for latent_ens_member_npy in latent_ens_members_npy:
    difference_mean = latent_ens_member_npy - latent_center_npy
    difference_mean = np.reshape(difference_mean, (autoencoder.latent_dim))
    matrices.append(np.tensordot(difference_mean, difference_mean, axes=0)) # tensor product
B_matrix = np.mean(matrices, axis=0)
pickle.dump(B_matrix,
            open(f'decoding_experiment013--Fully_flow_dependent_B-matrix_from_ERA5_ensemble_members-data/de013--Fully_flow_dependent_B-matrix_from_ERA5_ensemble_members-epoch={args.epoch}_{args.year:04d}_{args.month:02d}_{args.year:02d}_center={args.center_option}'
            + '_matrix.pkl', 'wb'))


if args.plot:
    from matplotlib.colors import LogNorm

    plt.figure(figsize=(12, 12))
    #matplotlib.rcParams.update({"font.size": 12})
    mat = plt.matshow(np.abs(B_matrix), norm=LogNorm(vmin=1e-4, vmax=1e-2), cmap='gist_earth_r')
    plt.colorbar(mat, fraction=0.092, pad=0.03, shrink=0.85, extend='both')
    plt.title(r'abs($\mathbf{B}_z$) from ERA5 ensemble members')
    #plt.tight_layout()
    plt.savefig(
    f'decoding_experiment013--Fully_flow_dependent_B-matrix_from_ERA5_ensemble_members-figures/de013--Fully_flow_dependent_B-matrix_from_ERA5_ensemble_members-epoch={args.epoch}_{args.year:04d}_{args.month:02d}_{args.day:02d}_center={args.center_option}'
        + '.jpg', dpi=300)
    print('done')

    print('min diag. element', np.amin(np.diagonal(B_matrix)))
    print('max diag. element', np.amax(np.diagonal(B_matrix)))
    print('max offdiag. element', np.amax(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
    print('mean diag. element', np.mean(np.diagonal(B_matrix)))
    print('mean offdiag. element', np.mean(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
    print('ratio of means',
          np.mean(np.diagonal(B_matrix)) / np.mean(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
    print('sum diag. elements / sum offdiag. elements',
          np.sum(np.diagonal(B_matrix)) / np.sum(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
    max_offdiag_values = np.max(np.abs((B_matrix - B_matrix * np.identity(latent_dim))), axis=1)
    print('number of elements with larger offdiag. than diag. value',
          np.sum(np.where(np.diagonal(B_matrix) < max_offdiag_values, 1, 0)))
    print('worst ratio between diag. and offdiag. value', np.max(max_offdiag_values / np.diagonal(B_matrix)))

    plt.cla()
    plt.clf()
    plt.figure(figsize=(4*1.05*1.017, 4*1.05*1.017))

    nbin = 16#21
    bins = np.linspace(-4,-1, nbin)
    plt.hist(np.log10(np.diagonal(B_matrix)), bins=bins, density=True, alpha=0.8, label='Diagonal elements')
    plt.hist(np.log10(np.abs(B_matrix - B_matrix * np.identity(latent_dim)).flatten()), bins=bins, density=True, alpha=0.8, label='Off-diagonal elements')
    plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z$ element))')
    plt.xlim(min(bins), max(bins))
    plt.ylabel('Percentage')
    plt.legend()
    plt.title(r'Distribution of $\mathbf{B}_z$ elements')
    plt.yticks(ticks=nbin/(max(bins) - min(bins)) * np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), labels=['0', '20', '40', '60', '80', '100']) # to mapiranje postudiraj
    plt.tight_layout()
    plt.savefig(f'decoding_experiment013--Fully_flow_dependent_B-matrix_from_ERA5_ensemble_members-figures/de013--Fully_flow_dependent_B-matrix_from_ERA5_ensemble_members-epoch={args.epoch}_{args.year:04d}_{args.month:02d}_{args.day:02d}_center={args.center_option}'
                '_hist.jpg', dpi=300)


# Testing distribution hypothesis
N = 9
samples = np.random.normal(size=N)
intermediates = [(samples[i+1] + samples[i])/2 for i in range(len(samples)-1)]