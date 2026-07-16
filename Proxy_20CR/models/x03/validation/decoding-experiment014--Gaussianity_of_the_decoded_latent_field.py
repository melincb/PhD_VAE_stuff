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
parser.add_argument("--compute", help='Simulate a large number of realisation in the grid point space from a perturbed latent vector.', default=False, action=argparse.BooleanOptionalAction)
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


N = 10**4
M = 100

if args.compute:
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


    latent_mean_orig, latent_logvar_orig = autoencoder.encode(t_in)

    sample_by_sample_latent = []
    sample_by_sample_gp = []

    for i in range(N//M):
        print(i)
        latent_samples = tf.Variable(tf.random.normal(mean=latent_mean_orig, stddev=np.sqrt(np.diagonal(B_matrix)), shape=(M, autoencoder.latent_dim)))
        decoded = autoencoder.decode(latent_samples).numpy().astype(np.float16)

        for j in range(M):
            sample_by_sample_latent.append(latent_samples[j].numpy().astype(np.float16))
            sample_by_sample_gp.append(decoded[j])

    pickle.dump(sample_by_sample_latent, open(f'decoding_experiment014--Gaussianity_of_the_decoded_latent_field-data/de014--{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{N}_latent_samples.pkl', 'wb'))
    pickle.dump(sample_by_sample_gp, open(f'decoding_experiment014--Gaussianity_of_the_decoded_latent_field-data/de014--{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{N}_gridpoint_samples.pkl', 'wb'))

sample_by_sample_latent = np.array(pickle.load(open(f'decoding_experiment014--Gaussianity_of_the_decoded_latent_field-data/de014--{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{N}_latent_samples.pkl', 'rb'))).astype(np.float16)
print(np.shape(sample_by_sample_latent))
all_skewness_latent = []
all_kurt_latent = []
import scipy
for i in range(np.shape(sample_by_sample_latent)[1]):
    all_skewness_latent.append(scipy.stats.skew(sample_by_sample_latent[:,i]))
    all_kurt_latent.append(scipy.stats.kurtosis(sample_by_sample_latent[:,i]))
print(np.mean(np.abs(all_skewness_latent)))
print(np.mean(np.abs(all_kurt_latent)))

sample_by_sample_gp = np.array(pickle.load(open(f'decoding_experiment014--Gaussianity_of_the_decoded_latent_field-data/de014--{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{N}_gridpoint_samples.pkl', 'rb'))).astype(np.float16)
print(np.shape(sample_by_sample_gp))
all_skewness_gp = []
all_kurt_gp = []
for i in range(np.shape(sample_by_sample_gp)[1]):
    if i%10 == 0:
        print(i)
    for j in range(np.shape(sample_by_sample_gp)[2]):
        sk = scipy.stats.skew(sample_by_sample_gp[:, i, j, 0]) # 0... only temperature field
        ku = scipy.stats.kurtosis(sample_by_sample_gp[:, i, j, 0])
        all_skewness_gp.append(sk)
        all_kurt_gp.append(ku)
    pickle.dump(all_skewness_gp[-np.shape(sample_by_sample_gp)[2]:], open(f'decoding_experiment014--Gaussianity_of_the_decoded_latent_field-data/sk_{i}.pkl', 'wb'))
    pickle.dump(all_kurt_gp[-np.shape(sample_by_sample_gp)[2]:], open(f'decoding_experiment014--Gaussianity_of_the_decoded_latent_field-data/ku_{i}.pkl', 'wb'))
print(np.mean(np.abs(all_skewness_gp)))
print(np.mean(np.abs(all_kurt_gp)))