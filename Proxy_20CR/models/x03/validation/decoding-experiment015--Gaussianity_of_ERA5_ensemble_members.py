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
parser.add_argument("--compute", help='Compute mean p-value for each date', default=False, action=argparse.BooleanOptionalAction)
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


latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % (1000*(args.epoch//1000))]
autoencoder = DCVAE(latent_dim=latent_dim)

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

dates = [
    datetime.date(2015, 1, 15),
    datetime.date(2015, 4, 15),
    datetime.date(2015, 7, 15),
    datetime.date(2015, 10, 15),
    datetime.date(2016, 1, 15),
    datetime.date(2016, 4, 15),
    datetime.date(2016, 7, 15),
    datetime.date(2016, 10, 15),
    datetime.date(2017, 1, 15),
    datetime.date(2017, 4, 15),
    datetime.date(2017, 7, 15),
    datetime.date(2017, 10, 15),
    datetime.date(2018, 1, 15),
    datetime.date(2018, 4, 15),
    datetime.date(2018, 7, 15),
    datetime.date(2018, 10, 15),
    datetime.date(2019, 1, 15),
    datetime.date(2019, 4, 15),
    datetime.date(2019, 7, 15),
    datetime.date(2019, 10, 15),
    datetime.date(2020, 1, 15),
    datetime.date(2020, 4, 15),
    datetime.date(2020, 7, 15),
    datetime.date(2020, 10, 15),
    datetime.date(2021, 1, 15),
    datetime.date(2021, 4, 15),
    datetime.date(2021, 7, 15),
    datetime.date(2021, 10, 15),
    datetime.date(2022, 1, 15),
    datetime.date(2022, 4, 15),
    datetime.date(2022, 7, 15),
    datetime.date(2022, 10, 15),
]

if args.compute:
    all_gp_shapiro = []
    all_gp_shapiro_p = []
    all_gp_shapiro_p_mean = []
    all_latent_shapiro = []
    all_latent_shapiro_p = []
    all_latent_shapiro_p_mean = []
    for date in dates:
        latent_vectors = []
        gridpoint_fields = []
        for iens_member in range(1,9+1):

            t = ERA5_load_T850(date.year, date.month, date.day, ensemble_member=iens_member)
            t_orig = t.copy()
            c = ERA5_load_T850_climatology(date.year, date.month, date.day)
            vc = ERA5_load_T850_variability_climatology(date.year, date.month, date.day)
            t = t - c
            t = t / vc
            t = ERA5_roll_longitude(t)
            # t = ERA5_trim(t)
            t_in = tf.convert_to_tensor(t.data, np.float32)
            t_in = tf.reshape(t_in, [1, 720, 1440, 1])

            vc = ERA5_roll_longitude(vc)
            vc = vc.data


            latent_mean_orig, latent_logvar_orig = autoencoder.encode(t_in)
            gridpoint_fields.append(np.squeeze(t_in.numpy()))
            latent_vectors.append(np.squeeze(latent_mean_orig.numpy()))

        # Shapiro test for each grid point
        gridpoint_fields = np.array(gridpoint_fields)
        latent_vectors = np.array(latent_vectors)
        for i in range(720):
            if i%100 == 0:
                print(date, i)
            for j in range(1440):
                shapiro, shapiro_p = scipy.stats.shapiro(gridpoint_fields[:, i, j])
                all_gp_shapiro.append(shapiro)
                all_gp_shapiro_p.append(shapiro_p)

        for i in range(np.shape(latent_vectors)[1]):
            shapiro, shapiro_p = scipy.stats.shapiro(latent_vectors[:, i])
            all_latent_shapiro.append(shapiro)
            all_latent_shapiro_p.append(shapiro_p)

        all_gp_shapiro_p_mean.append(np.mean(all_gp_shapiro_p[-720*1440:]))
        all_latent_shapiro_p_mean.append(np.mean(all_latent_shapiro_p[-latent_dim:]))
        print('gp', np.mean(all_gp_shapiro[-720*1440:]), np.mean(all_gp_shapiro_p[-720*1440:]))
        print('latent',np.mean(all_latent_shapiro[-latent_dim:]), np.mean(all_latent_shapiro_p[-latent_dim:]))

    pickle.dump([all_gp_shapiro_p_mean, all_latent_shapiro_p_mean], open('decoding_experiment015--Gaussianity_of_ERA5_ensemble_members-data/shapiro_p_gp_and_latent.pkl', 'wb'))

all_gp_shapiro_p_mean, all_latent_shapiro_p_mean = pickle.load(open('decoding_experiment015--Gaussianity_of_ERA5_ensemble_members-data/shapiro_p_gp_and_latent.pkl', 'rb'))
plt.scatter([i for i in dates], all_gp_shapiro_p_mean, marker='s', color='r', label='Original ens. members (grid point space)')
plt.scatter([i for i in dates], all_latent_shapiro_p_mean, marker='*', color='b', label='Encoded ens. members (latent space)')
plt.figure(figsize=(12, 5))
plt.scatter([i for i in range(len(dates))], all_gp_shapiro_p_mean, marker='s', color='r', label='Original ens. members (grid point space)')
plt.scatter([i for i in range(len(dates))], all_latent_shapiro_p_mean, marker='*', color='b', label='Encoded ens. members (latent space)')
plt.xticks(ticks=[i for i in range(len(dates))], labels=[d.strftime("%Y-%m-%d") for d in dates], rotation=90)
plt.grid(linestyle=':', linewidth=0.6)
plt.legend(loc='lower right')
plt.ylabel('p-value from Shapiro-Wilk test')
plt.tight_layout()
plt.savefig('decoding_experiment015--Gaussianity_of_ERA5_ensemble_members-data/shapiro_p_gp_and_latent.jpg', dpi=300)

print(np.mean(all_gp_shapiro_p_mean), np.std(all_gp_shapiro_p_mean), np.mean(all_latent_shapiro_p_mean), np.std(all_latent_shapiro_p_mean))