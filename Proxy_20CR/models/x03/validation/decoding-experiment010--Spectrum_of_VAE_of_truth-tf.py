#!/usr/bin/env python

# This program computes VAE of truth and stores it in .np format. It should be run in ProxyR virtual environment.
# The output then enters decoding-experiment010--Spectrum_of_VAE_of_truth-pyshtools.py, which computes its spectrum.
# The second program needs to be run in pyshtools virtual environment.

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
from plot_ERA5_comparison import get_land_mask  #this is actually here to get the path to ERA5_load

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
tnpy = np.array(t.data)
print('type tnpy', type(tnpy))
pickle.dump(tnpy,
            open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_TRUTH_field.pkl', 'wb'))
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




latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % (1000*(args.epoch//1000))]
autoencoder = DCVAE(latent_dim=latent_dim)


weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
    args.epoch,
)
print('Uploading weights from', weights_dir)
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

#latent = tf.Variable(constant * np.ones(shape=(1, autoencoder.latent_dim)))    # Bostjan: For constant values of all members of latent space
#fitted = autoencoder.sample_call(t_in, size=args.ensemble)  # TO SE NI OK
latent_mean_orig, latent_logvar_orig = autoencoder.encode(t_in)
#print('Encoded')
latent_sigma_orig = tf.exp(latent_logvar_orig * 0.5)
latent_samples = tf.Variable(tf.random.normal(mean=latent_mean_orig, stddev=latent_sigma_orig, shape=(args.ensemble, autoencoder.latent_dim)))
#print('Sampled')
decoded = autoencoder.decode(latent_samples)
#print('Decoded')

decoded = decoded.numpy()
decoded_renormalized = np.reshape(decoded, (args.ensemble, 720, 1440)) * vc
print(np.shape(decoded_renormalized))
#raise AssertionError
pickle.dump(decoded_renormalized,
            open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch={args.epoch:04d}_ensemble={args.ensemble}_decoded_fields.pkl', 'wb'))