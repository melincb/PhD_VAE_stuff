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
import datetime

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import argparse
import pickle


start = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=1020)
parser.add_argument(
    "--ensemble", help="No. of ensemble members", type=int, required=False, default=50
)
parser.add_argument("--year", help="Year", type=int, required=False, default=2019)
parser.add_argument(
    "--month", help="Integer month", type=int, required=False, default=4
)
parser.add_argument("--day", help="Day of month", type=int, required=False, default=15)
parser.add_argument("--oyear", help="Year", type=int, required=False)
parser.add_argument("--omonth", help="Integer month", type=int, required=False)
parser.add_argument("--oday", help="Day of month", type=int, required=False)
parser.add_argument(
    "--osize", help="Obs. point size", type=float, required=False, default=1.0
)
parser.add_argument('--compute', help="Compute assimilation", default=False, action=argparse.BooleanOptionalAction)  #In order not to compute: --no-compute
parser.add_argument('--plot', help="Plot", default=False, action=argparse.BooleanOptionalAction) #In order not to plot: --no-plot
parser.add_argument('--std_first_multiplier', help="Multiplier of std of first guess", type=float, required=False, default=1.0)
parser.add_argument('--obs_std', help="Standard deviation of pseudo observations, degree C", type=float, required=False, default=0.0)
parser.add_argument('--minimization_learning_rate', help='Learning rate for ADAM optimizer when performing minimization in latent space', type=float, required=False, default=0.01)
parser.add_argument('--adaptive_lr', help='Whether the learning rate for ADAM optimizer when performing minimization in latent space decreases if loss is on plateau or not', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--perfect_obs', help='Perfect observations', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--perfect_first', help='Perfect first guess', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--save_as_pdf', help='Save final figure also in pdf format', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--custom_addon', type=str, default='', required=False)
parser.add_argument('--cpus', type=int, default=1, required=False)
parser.add_argument('--obs_increment', help="Observation increment for single observation experiment (if not 0.0, else the value is sampled from the 'truth' field)", type=float, required=False, default=0.0)
parser.add_argument('--singobs_lat', help="Latitude in case of single observation experiment", type=float, required=False, default=False)
parser.add_argument('--singobs_lon', help="Latitude in case of single observation experiment", type=float, required=False, default=False)
parser.add_argument('--diagonal_B', help='Only use diagonal elements of B-matrix (no correlations between latent elements)', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--flow_B_fraction', help="Fraction of B-matrix, computed from ensemble members of ERA5", type=float, required=False, default=0.0)
parser.add_argument('--ERA5_ens_mem_spread_multiplicator', help="The multiplicator of the ensemble spread from ERA5 ensemble members", type=int, required=False, default=1)
parser.add_argument('--plot_singles', help="Plot some figures one by one (only if --plot)", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--background_latent_mean_field_file', help="Filename for file, from which we extract the mean of each latent vector element, which is then used as the mean background. The background error is sampled from the (constant) B-matrix. If not specified, the mean encoded field from the previous day is used as the mean background", type=str, default='', required=False)
parser.add_argument('--cycling_starting_point', help='Date for the first cycle', type=str, default='', required=False)



args = parser.parse_args()
if args.oyear is None:
    args.oyear = args.year
if args.omonth is None:
    args.omonth = args.month
if args.oday is None:
    args.oday = args.day

check_if_in_test_set_path = '%s/Proxy_20CR/datasets/ERA5/daily_T850/regridded_version/x03test/%04d-%02d-%02d.tfd' % (os.getenv("SCRATCH"), args.year, args.month, args.day)
if os.path.isfile(check_if_in_test_set_path):
    print('\nChosen date is in the TEST set!\n')
else:
    check_if_in_validation_set_path = '%s/Proxy_20CR/datasets/ERA5/daily_T850/regridded_version/x03validation/%04d-%02d-%02d.tfd' % (os.getenv("SCRATCH"), args.year, args.month, args.day)
    if os.path.isfile(check_if_in_validation_set_path):
        print('\nChosen date is in the VALIDATION set!\n')
    else:
        print('\nChosen date IS NOT in the validation or test set\n')

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

init_lr = args.minimization_learning_rate



res = 4.0   # distance between observation points (or 'custom')
sym = False#True  # True: lats go from 90 - res/2 to -90 + res/2 (similarly lons)
            # False: lats go from 89.875 to -90 + res - 0.125 (similarly lons)
if (args.singobs_lon is not False) or (args.singobs_lat is not False):
    if (args.singobs_lon is not False) and (args.singobs_lat is not False):
        res = 'custom'
    else:
        raise AttributeError # You specified either singobs_lon or singobs_lat, but you should either specify both of them or none of them!
double_res = False#True # whether we also add the same grid, but diagonaly shifted
fake_zeros = False  # if true, the climatological mean is used as the input
res_str = str(res)
if double_res:
    res_str += 'd'
if sym:
    res_str += 's'


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )

print('PREPARING DATA!')
this_day = datetime.date(args.year, args.month, args.day)
t = ERA5_load_T850(args.year, args.month, args.day)
c = ERA5_load_T850_climatology(args.year, args.month, args.day)
vc = ERA5_load_T850_variability_climatology(args.year, args.month, args.day)
t = t - c
if fake_zeros:
    t = t/t*1 # same effect as if t = c from the start
t = t / vc
t = ERA5_roll_longitude(t)
t_in = tf.convert_to_tensor(t.data, np.float32)
t_in = tf.reshape(t_in, [1, 720, 1440, 1])
vc = ERA5_roll_longitude(vc)
vc = vc.data

true_mean, true_logvar = autoencoder.encode(t_in)
true_mean = true_mean.numpy().reshape(autoencoder.latent_dim)
true_logvar = true_logvar.numpy().reshape(autoencoder.latent_dim)
true_std = np.sqrt(np.exp(true_logvar))
true_half_iqr = true_std * 0.6744  # Bostjan: experimented with scipy.stats.norm.cdf, also in Bronstein (ish)

if args.background_latent_mean_field_file == '':
    previous_day = this_day - datetime.timedelta(days=1)
    t_previous = ERA5_load_T850(previous_day.year, previous_day.month, previous_day.day)
    c_previous = ERA5_load_T850_climatology(previous_day.year, previous_day.month, previous_day.day)
    vc_previous = ERA5_load_T850_variability_climatology(previous_day.year, previous_day.month, previous_day.day)
    t_previous = t_previous - c_previous
    if fake_zeros:
        t_previous = t_previous / t_previous * 0
    t_previous = t_previous / vc_previous
    t_previous = ERA5_roll_longitude(t_previous)
    t_previous_in = tf.convert_to_tensor(t_previous.data, np.float32)
    t_previous_in = tf.reshape(t_previous_in, [1, 720, 1440, 1])
    vc_previous = ERA5_roll_longitude(vc_previous)
    vc_previous = vc_previous.data

    previous_true_mean, previous_true_logvar = autoencoder.encode(t_previous_in)
    background_latent_mean = previous_true_mean.numpy().reshape(autoencoder.latent_dim)
    previous_true_logvar = previous_true_logvar.numpy().reshape(autoencoder.latent_dim)
    previous_true_std = np.sqrt(np.exp(previous_true_logvar))
    previous_true_half_iqr = previous_true_std * 0.6744
else:
    file_to_load = args.background_latent_mean_field_file

    dict_to_load = pickle.load(open(file_to_load + '.pkl', 'rb'))
    latent = dict_to_load['latent']
    analysis_latent_mean_from_previous = np.mean(latent, axis=0)
    background_latent_mean = analysis_latent_mean_from_previous.reshape(autoencoder.latent_dim)





if args.compute:
    print('COMPUTING!')

        # Get the ob locations at the given time from 20CRv3
    dte = datetime.datetime(args.oyear, args.omonth, args.oday, 12)
    #obs = twcr.load_observations_1file(dte, version="3")
    # print(obs["Latitude"])
    # print(obs["Longitude"])
    # print(type(obs["Latitude"].values))
    # print(obs["Latitude"].values)
    if type(res) == float:
        if sym:
            lats = np.array(
                [[90 - res/2 - res * ilat for ilon in range(int(360 / res))] for ilat in range(int(180 / res))])
            lons = np.array(
                [[-180 + res/2 + res * ilon for ilon in range(int(360 / res))] for ilat in range(int(180 / res))])
        else:
            lats = np.array([[90 - 0.125 - res*ilat for ilon in range(int(360/res))] for ilat in range(int(180/res))])
            lons = np.array([[-180 + 0.125 + res*ilon for ilon in range(int(360/res))] for ilat in range(int(180/res))])
        if args.custom_addon == 'west_only':
            lats = np.array([[90 - 0.125 - res * ilat for ilon in range(int(180 / res))] for ilat in range(int(180 / res))])
            lons = np.array([[-180 + 0.125 + res * ilon for ilon in range(int(180 / res))] for ilat in range(int(180 / res))])
        if double_res:
            lats = np.concatenate((lats, lats-res/2), axis=0)
            lons = np.concatenate((lons, lons+res/2), axis=0)
            #print(lats)
    else:   # 'custom'
        # lats = np.array([[30 for i in range(10)]])#
        # lons = np.array([[-100 + i for i in range(10)]])#
        # Ljubljana: (46.056946, 14.505751), Jakarta(-6.200000, 106.816666), Singapore (1.290270, 103.851959)
        lats = np.array([[args.singobs_lat]])
        lons = np.array([[args.singobs_lon]])
    # print(np.amin(lats), np.amax(lats), np.amin(lons), np.amax(lons))
    # input('END')
    # Convert the obs locations to a tensor in the right units (0-1)
    print('lats', lats)
    t_lats_npy = (lats.flatten() + 90)/180 # orig. (obs["Latitude"].values + 90) / 180
    t_lons_npy = (lons.flatten() + 180)/360 # orig. (obs["Longitude"].values) / 360
    print('t_lats', t_lats_npy)
    print('lons', lons)
    print('t_lons_npy', t_lons_npy)

    # t_lons[t_lons > 0.5] -= 1 # zakomentirano - stestirano, da je neustrezno
    # t_lons += 0.5 # zakomentirano - stestirano, da je neustrezno
    t_lats = tf.convert_to_tensor(1.0 - t_lats_npy, tf.float32) # ker so do zdaj t_lats=180 na NP in t_lats=0 na SP,
                                                            # medtem ko so idx v matrikah 0 na NP in 719 na SP
    t_lons = tf.convert_to_tensor(t_lons_npy, tf.float32)
    #print(np.amin(t_lats), np.amax(t_lats), np.amin(t_lons), np.amax(t_lons))
    print('t_lats', np.shape(t_lats))
    #input('end3')
    t_obs = tf.stack((t_lats * 720, t_lons * 1440), axis=1)
    print('t_obs', t_obs)
    #time.sleep(60)
    t_obs = tf.expand_dims(t_obs, 0)
    print('t_obs', np.shape(t_obs))

    #if np.shape(lats) != (1,1):
    #    print('bilinear', np.shape(interpolate_bilinear(t_in, t_obs, indexing="ij")))
    if args.obs_increment == 0.0:
        # Normal experiment
        exact = tf.squeeze(interpolate_bilinear(t_in, t_obs, indexing="ij"), [0, 2])
    else:
        # Observation increment
        background_gp_control = autoencoder.decode(background_latent_mean.reshape(1,autoencoder.latent_dim)) * vc.reshape(np.shape(t_in)) # NOT * vc_previous
        exact = tf.squeeze(interpolate_bilinear(background_gp_control, t_obs, indexing="ij") + args.obs_increment, [0, 2])
    # else:
    #     exact = interpolate_bilinear(t_in, t_obs, indexing="ij")
    print('exact', np.shape(exact))
    #print('bilinear vc', np.shape(interpolate_bilinear(vc.reshape(np.shape(t_in)), t_obs, indexing="ij")))
    vc_obs = tf.squeeze(interpolate_bilinear(vc.reshape(np.shape(t_in)), t_obs, indexing="ij"), [0, 2])
    # Filter out the nans (bad lat/lon)
    #if np.shape(lats) != (1,1): # otherwise there is always an error
    t_obs = tf.boolean_mask(t_obs, ~tf.math.is_nan(exact), axis=1)
    exact = tf.boolean_mask(exact, ~tf.math.is_nan(exact), axis=0)
    vc_obs = tf.boolean_mask(vc_obs, ~tf.math.is_nan(exact), axis=0)
    # print(type(exact))




    # # Make a set of fitted fields
    # fitted = []
    # latents = []
    if args.obs_increment == 0:
        exact_renormalized = exact * vc_obs
    else:
        exact_renormalized = exact  # to make it simpler when adding the observation increment
    print('exact, exact_renormalized', exact, exact_renormalized)
    R_matrix = np.identity(exact.shape[0]) * args.obs_std**2
    #B_matrix = np.identity(autoencoder.latent_dim) * previous_true_std**2 + np.random.normal(size=(autoencoder.latent_dim, autoencoder.latent_dim)) * 1e-4
    B_matrix = pickle.load(open('../validation/decoding_experiment006---B-matrix_for_persistence-data/B_matrix_2015-01-01_to_2018-12-31.pkl', 'rb'))
    if args.flow_B_fraction > 0:
        print('getting control')
        t_control = ERA5_load_T850(args.year, args.month, args.day, ensemble_member='control')
        t_control = t_control - c
        t_control = ERA5_roll_longitude(t_control)
        t_control_origish = t_control.copy()
        t_control = t_control / vc
        t_control_in = tf.convert_to_tensor(t_control.data, np.float32)
        t_control_in = tf.reshape(t_control_in, [1, 720, 1440, 1])
        latent_control_mean, latent_control_logvar = autoencoder.encode(t_control_in)
        t_ens_members = []
        true_spread_multiplicator = args.ERA5_ens_mem_spread_multiplicator
        for iens_member in range(1, 9+1):
            print('ens member', iens_member)
            t_iens_member = ERA5_load_T850(args.year, args.month, args.day, ensemble_member=iens_member)
            t_iens_member = t_iens_member - c
            t_iens_member = ERA5_roll_longitude(t_iens_member)
            t_iens_member = true_spread_multiplicator * (t_iens_member.data - t_control_origish.data) + t_control_origish.data
            t_iens_member = t_iens_member / vc
            t_ens_members.append(t_iens_member)
        t_ens_members_in = tf.convert_to_tensor(t_ens_members, np.float32)
        t_ens_members_in = tf.reshape(t_ens_members_in, [9, 720, 1440, 1])
        # ens_members_spread = tf.squeeze(tf.math.reduce_std(t_ens_members_in, axis=0)) * vc
        # lm = get_land_mask()
        # fig2 = plt.figure(figsize=(6, 4))
        # ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        # lats = lm.coord("latitude").points
        # lons = lm.coord("longitude").points
        # lons, lats = np.meshgrid(lons, lats)
        # print('entering pc')
        # pc = ax2.pcolormesh(lons, lats, ens_members_spread, transform=ccrs.PlateCarree(),
        #                     cmap='terrain_r')
        # print('adding coastlines')
        # ax2.coastlines()
        # print('setting global')
        # ax2.set_global()
        # fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05)
        # ax2.set_title(f'{true_spread_multiplicator} x ERA5 ensemble members std', y=1.02)
        # gl = ax2.gridlines(
        #     draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        # )
        # print('saving')
        # fig2.savefig(
        #     'fit_multi-experiment005--Ensemble-3D-Var-figures/fe005--True_ensemble_members_%dx_spread_mutliplied-%04d-%02d-%02d'
        #     % (true_spread_multiplicator, args.year, args.month, args.day) + '.jpg', dpi=300)
        # fig2.savefig(
        #     'fit_multi-experiment005--Ensemble-3D-Var-figures/fe005--True_ensemble_members_spread-%04d-%02d-%02d'
        #     % (args.year, args.month, args.day) + '.pdf')


        latent_ens_members_mean, latent_ens_members_logvar = autoencoder.encode(t_ens_members_in)
        print('Got all latent!')
        print('shape latent_control', np.shape(latent_control_mean.numpy()))
        print('shape latent_ens_members', np.shape(latent_ens_members_mean.numpy()))
        latent_control_npy = latent_control_mean.numpy()
        latent_ens_members_npy = latent_ens_members_mean.numpy()
        matrices = []
        for latent_ens_member_npy in latent_ens_members_npy:
            difference_mean = latent_ens_member_npy - latent_control_npy
            difference_mean = np.reshape(difference_mean, (autoencoder.latent_dim))
            matrices.append(np.tensordot(difference_mean, difference_mean, axes=0)) # tensor product
        B_matrix_flow = np.mean(matrices, axis=0)
        B_matrix = (1 - args.flow_B_fraction) * B_matrix + args.flow_B_fraction * B_matrix_flow
        #print(np.shape(B_matrix_flow))
        #print(np.diagonal(B_matrix_flow))
        #print(B_matrix_flow[0])

        # from matplotlib.colors import LogNorm
        #
        # plt.figure(figsize=(12, 12))
        # mat = plt.matshow(np.abs(B_matrix_flow), norm=LogNorm(vmin=1e-4, vmax=1), cmap='gist_earth_r')
        # plt.colorbar(mat, fraction=0.092, pad=0.03, shrink=0.85)
        # plt.title(r'Flow dependent part of B-matrix for date %04d-%02d-%02d' % (args.year, args.month, args.day))
        # # plt.tight_layout()
        # plt.savefig(
        #     'fit_multi-experiment005--Ensemble-3D-Var-figures/fe005--Flow_dependent_part_of_B-matrix_%dx_spread_mutliplied-%04d-%02d-%02d'
        #     % (true_spread_multiplicator, args.year, args.month, args.day) +'.jpg', dpi=300)
        # plt.savefig(
        #     'fit_multi-experiment005--Ensemble-3D-Var-figures/fe005--Flow_dependent_part_of_B-matrix_%dx_spread_mutliplied-%04d-%02d-%02d'
        #     % (true_spread_multiplicator, args.year, args.month, args.day) +'.pdf')
        #
        # raise AssertionError


    if args.diagonal_B:
        B_matrix = np.diagonal(B_matrix) * np.identity(len(B_matrix))
    R_matrix_inv = np.linalg.inv(R_matrix)#tf.convert_to_tensor(np.linalg.inv(R_matrix), dtype=tf.float32)
    B_matrix_inv = np.linalg.inv(B_matrix)#tf.convert_to_tensor(np.linalg.inv(B_matrix), dtype=tf.float32)

    print('sqrt(diag(B))', np.sqrt(np.diagonal(B_matrix)))
    print('ensemble member')
    #if args.perfect_first:
    previous_latent = tf.constant(tf.random.normal(shape=(args.ensemble, 1, autoencoder.latent_dim), mean=background_latent_mean, stddev=np.sqrt(np.diagonal(B_matrix))))
    previous_latent = np.float32(
        np.random.normal(size=(args.ensemble, 1, autoencoder.latent_dim), loc=background_latent_mean,
                         scale=np.sqrt(np.diagonal(B_matrix))
                         ))
    # # If you want to use the very same ensemble of background vectors:
    # previous_latent = pickle.load(open('fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_2019-4-15_epoch=1020_obs_std=1.0_res=custom_ensemble=150_diagonal_B_singobs_Ljubljana+3K.pkl', 'rb'))['previous_latent'].numpy()
    # previous_latent = np.reshape(previous_latent, (args.ensemble, 1, autoencoder.latent_dim))
    # previous_latent = np.float32(
    #     np.random.normal(size=(args.ensemble, 1, autoencoder.latent_dim), loc=background_latent_mean,
    #                      scale=0.0))

    print('got previous_latent')
    # print(previous_latent)

    #else:
    #    AssertionError # Imperfect first guess not yet available!
    #latent = previous_latent   # otherwise it will change previous_latent to latent
    #latent = tf.Variable(tf.identity(latent))
    #previous_latents.append(tf.identity(previous_latent))
    #if args.perfect_obs:
    #    pseudo_obs = tf.random.normal(
    #        shape=(args.ensemble, exact_renormalized.shape[0]), mean=exact_renormalized[0], stddev=0.0, dtype=tf.float32
    #    )
    #else:
    # print('exact_renormalized', exact_renormalized)
    # print('exact_renormalized[0]', exact_renormalized[0])
    pseudo_obs = tf.random.normal(
            shape=(args.ensemble, exact_renormalized.shape[0], 1), mean=exact_renormalized, stddev=args.obs_std, dtype=tf.float32
        )
    pseudo_obs = np.float32(np.random.normal(
        size=(args.ensemble, exact_renormalized.shape[0], 1), loc=np.reshape(exact_renormalized, (1, exact_renormalized.shape[0], 1)), scale=args.obs_std
    ))
    print('exact renormalized', exact_renormalized)
    # # If you want to use the very same ensemble of pseudo observations:
    # pseudo_obs = pickle.load(open('fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_2019-4-15_epoch=1020_obs_std=1.0_res=custom_ensemble=150_diagonal_B_singobs_Ljubljana+3K.pkl', 'rb'))['pseudo_obs'].numpy()
    # pseudo_obs = np.reshape(pseudo_obs, (args.ensemble, exact_renormalized.shape[0], 1))
    # pseudo_obs = np.float32(np.random.normal(
    #     size=(args.ensemble, exact_renormalized.shape[0], 1), loc=np.reshape(exact_renormalized, (1, exact_renormalized.shape[0], 1)), scale=0.0
    # ))

    print('got pseudo_obs')
    # print(pseudo_obs)
    # time.sleep(60)
    #pseudo_obs = tf.reshape(pseudo_obs, (exact_renormalized.shape[0], 1))
    #print(np.shape(latent), np.shape(previous_latents[-1]))
    #latent = previous_latents[-1]

    # del autoencoder
    # gc.collect()

    inputs_for_ensemble_3D_Var = [{'latent_background': previous_latent[iens_mem],
                'ob_locations':[t_lats_npy, t_lons_npy],
                'pseudo_obs':pseudo_obs[iens_mem],
                'renormalization':vc,
                'B_matrix_inv':B_matrix_inv,
                'R_matrix_inv':R_matrix_inv,
                'init_lr':init_lr,
                'epoch':args.epoch,
                'ensemble_member_idx':iens_mem} for iens_mem in range(args.ensemble)]

    obs_gp_std = [args.obs_std for ob in exact_renormalized]

    file_to_dump = f'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_{args.year}-{args.month}-{args.day}_epoch={args.epoch}_obs_std={args.obs_std}_res={res_str}_ensemble={args.ensemble}'
    if args.std_first_multiplier != 1.0:
        file_to_dump += f'std_first_mutiplier={args.std_first_multiplier}'
    if args.minimization_learning_rate != 0.01:
        file_to_dump += f'minimization_lr={args.minimization_learning_rate}'
    if not args.adaptive_lr:
        file_to_dump += '_no_adaptive_lr'
    if args.perfect_first:
        file_to_dump += '_perfect_first'
    if args.perfect_obs:
        file_to_dump += '_perfect_obs'
    if args.diagonal_B:
        file_to_dump += '_diagonal_B'
    if args.flow_B_fraction > 0:
        file_to_dump += f'_flow_B_fraction={args.flow_B_fraction}'
        if args.ERA5_ens_mem_spread_multiplicator != 1:
            file_to_dump += f'_ERA5_ens_spr_mul={args.ERA5_ens_mem_spread_multiplicator}'
    if args.cycling_starting_point != '':
        file_to_dump += f'_cycling_from_{args.cycling_starting_point}'
    if len(args.custom_addon) > 0:
        file_to_dump += '_' + args.custom_addon

    dict_to_dump = {'inputs_for_ensemble_3D_Var':inputs_for_ensemble_3D_Var,
                    'cpus':args.cpus,
                    'name':file_to_dump,
                    'obs_gp_std':obs_gp_std}
    pickle.dump(dict_to_dump, open('fit_multi-experiment005--Ensemble-3D-Var-data/algorithm_inputs.pkl', 'wb'))

    del autoencoder
    gc.collect()

    # inputs = [{'latent_background': previous_latent[iens_mem],
    #            'ob_locations':t_obs,
    #             'pseudo_obs':pseudo_obs[iens_mem],
    #             'renormalization':vc,
    #             'B_matrix_inv':B_matrix_inv,
    #             'R_matrix_inv':R_matrix_inv,
    #             'init_lr':init_lr} for iens_mem in range(args.ensemble)]
    # print('got inputs')
    # pool = multiprocessing.Pool(processes=args.cpus)
    # print('defined pool')
    # results = pool.map(parallel_worker, inputs)
    # print('results!')
    # pool.close()
    # pool.join()
    #
    # latent = np.array([results[iens_mem]['best_latent'] for iens_mem in range(args.ensemble)])
    # best_loss = [results[iens_mem]['best_loss'] for iens_mem in range(args.ensemble)]
    # losses = [results[iens_mem]['losses'] for iens_mem in range(args.ensemble)]
    # logpzs = [results[iens_mem]['logpzs'] for iens_mem in range(args.ensemble)]
    #
    # the_3DVar_outputs = (latent, best_loss, losses, logpzs)
    # # latents.append(tf.identity(latent[0]))
    # # #fitted.append(autoencoder.decode(tf.identity(latent)))
    # # #fitted = tf.stack(fitted, axis=1)[0]
    # # #print(np.shape(fitted))
    # # #print('latents', latents)
    # # latent = tf.Variable(latents)
    # # previous_latents_shape = np.shape(previous_latents)
    # # previous_latents = np.reshape(previous_latents, (previous_latents_shape[0], previous_latents_shape[2])) # otherwise it's (previous_latents_shape[0], 1, previous_latents_shape[2])
    # # previous_latents = tf.Variable(previous_latents)
    #
    # # previous_latent = tf.Variable(tf.random.normal(shape=(args.ensemble, autoencoder.latent_dim), mean=background_latent_mean, stddev=args.std_first_multiplier*previous_true_std))
    # # latent = previous_latent   # otherwise it will change previous_latent to latent
    # # previous_latent = tf.identity(previous_latent)
    # # pseudo_obs_sample = exact + tf.random.normal(
    # #     shape=exact.shape, mean=0.0, stddev=args.noise / 15, dtype=tf.float32
    # # )
    # # (latent, loss) = findLatent(autoencoder, latent, t_obs, pseudo_obs_sample)
    # # print('Found latent')
    # # fitted = autoencoder.decode(latent)
    # # print(np.shape(fitted))
    #
    #
    #
    # comment = "latent: latent element values after fit (initial guess was sampled from true distribution of latent values of previous day, i.e. previous_latent)\n" + \
    #             "t_obs: observations' locations\n" + \
    #             "previous_latent: samples, used as initial guess for fit\n" + \
    #             "obs_gp_std: standard deviation of observations\n" + \
    #             "the_3DVar_outputs: outputs from 3D-Var\n"
    #             #"fitted: autoencoder.decode(latent)\n" + \
    #             # "decoded_truth: VAE(truth), i.e. autoencoder.decode(tf.Variable(tf.random.normal(mean=true_mean, stddev=true_std, shape=(args.ensemble, autoencoder.latent_dim))))\n" + \
    #             ##"t_in: normalized input from ERA5 for the day of interest\n" + \
    #             ##"t_previous_in: normalized input from ERA5 for the previous day\n" + \
    #             ##"true_mean: autoencoder.encode(t_in)[0]\n" + \
    #             ##"true_std: recalculated true_logvar=autoencoder.encode(t_in)[1]\n" + \
    #             ##"previous_true_mean: autoencoder.encode(t_previous_in)[0]\n" + \
    #             ##"previous_true_std: autoencoder.encode(t_previous_in)[1]\n" + \
    #             ##"vc: variability cliamtology for the day of interest (for renormalization of t_in)\n" + \
    #             ##"vc_previous: variability cliamtology for the day before the day of interest (for renormalization of t_previous_in)\n" + \
    #             #"background_gp_std: standard deviation of background, recalculated to gridpoint space\n" + \
    # dict_to_dump = {'latent':latent, 'fitted':fitted, 't_obs':t_obs,
    #                 'previous_latent':previous_latents, 'obs_gp_std':obs_gp_std,
    #                 'the_3DVar_outputs':the_3DVar_outputs,
    #                 'comment':comment}
    # #file_to_dump = f'fit_multi-experiment004--3D-Var_obs_on_regular_grid-data/fe003--Fit_pseudo_obs_on_quasi_regular_grid-data_{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}_obs_std={args.obs_std}_res={res}'
    #
    # pickle.dump(dict_to_dump, open(file_to_dump + '.pkl', 'wb'))
    # # fitted.append(autoencoder.decode(latent))
    #
    # # fitted = tf.stack(fitted, axis=0)

elif args.plot:
    print('PLOTTING!')
    if not args.compute:
        #file_to_load = f'fit_multi-experiment004--3D-Var_obs_on_regular_grid-data/fe003--Fit_pseudo_obs_on_quasi_regular_grid-data_{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}_obs_std={args.obs_std}_res={res}'
        file_to_load = f'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-data_{args.year}-{args.month}-{args.day}_epoch={args.epoch}_obs_std={args.obs_std}_res={res_str}_ensemble={args.ensemble}'
        if args.std_first_multiplier != 1.0:
            file_to_load += f'std_first_mutiplier={args.std_first_multiplier}'
        if args.minimization_learning_rate != 0.01:
            file_to_load += f'minimization_lr={args.minimization_learning_rate}'
        if not args.adaptive_lr:
            file_to_load += '_no_adaptive_lr'
        if args.perfect_first:
            file_to_load += '_perfect_first'
        if args.perfect_obs:
            file_to_load += '_perfect_obs'
        if args.diagonal_B:
            file_to_load += '_diagonal_B'
        if args.flow_B_fraction > 0:
            file_to_load += f'_flow_B_fraction={args.flow_B_fraction}'
            if args.ERA5_ens_mem_spread_multiplicator != 1:
                file_to_load += f'_ERA5_ens_spr_mul={args.ERA5_ens_mem_spread_multiplicator}'
        if args.cycling_starting_point != '':
            file_to_load += f'_cycling_from_{args.cycling_starting_point}'
        if len(args.custom_addon) > 0:
            file_to_load += '_' + args.custom_addon

        dict_to_load = pickle.load(open(file_to_load + '.pkl', 'rb'))
        latent = dict_to_load['latent']
        print('shape of latent', np.shape(latent.numpy()))
        # fitted = dict_to_load['fitted']
        # t_in = dict_to_load['t_in']
        # t_previous_in = dict_to_load['t_previous_in']
        t_obs = dict_to_load['t_obs']
        print('These are t_obs', t_obs)
        # true_mean = dict_to_load['true_mean']
        # true_std = dict_to_load['true_std']
        # decoded_truth = dict_to_load['decoded_truth']
        previous_latents = dict_to_load['previous_latent']
        pseudo_obs = dict_to_load['pseudo_obs']
        print(np.shape(pseudo_obs))
        # vc = dict_to_load['vc']
        # vc_previous = dict_to_load['vc_previous']
        # previous_true_mean = dict_to_load['previous_true_mean']
        # previous_true_std = dict_to_load['previous_true_std']
        # background_gp_std = dict_to_load['background_gp_std']
        obs_gp_std = dict_to_load['obs_gp_std']
        # the_3DVar_outputs = dict_to_load['the_3DVar_outputs']
        best_loss = dict_to_load['best_loss']
        loss = dict_to_load['loss']
        logpzs = dict_to_load['logpzs']
        all_Jo = dict_to_load['all_Jo']
        all_gradient = dict_to_load['all_gradients']

        comment = dict_to_load['comment']
        print(comment)

        lnpy = latent.numpy()

        # This always loads the same B matrix!
        B_matrix = pickle.load(open(
            '../validation/decoding_experiment006---B-matrix_for_persistence-data/B_matrix_2015-01-01_to_2018-12-31.pkl',
            'rb'))

        #print(lnpy[:,0])

    # Compute standard deviation of the background (propagated to the gridpoint space)
    background_gp = tf.squeeze(autoencoder.decode(previous_latents)) * vc
    background_gp_std = tf.math.reduce_std(background_gp, axis=0).numpy()
    background_gp_mean = tf.math.reduce_mean(background_gp, axis=0).numpy()
    # background_gp_normalized = autoencoder.decode(tf.random.normal(shape=(args.ensemble, autoencoder.latent_dim), mean=previous_true_mean,
    #                                                                     stddev=args.std_first_multiplier * previous_true_std))
    # background_gp_normalized_std = tf.math.reduce_std(background_gp_normalized, axis=0)
    # background_gp_std = tf.squeeze(background_gp_normalized_std).numpy() * vc_previous
    #print(np.shape(background_gp_normalized_std), np.shape(background_gp_std))
    # background_gp_from_mean_normalized = autoencoder.decode(tf.random.normal(shape=(1, autoencoder.latent_dim), mean=previous_true_mean,
    #                                                                     stddev=0.0))
    # background_gp_from_mean = tf.squeeze(background_gp_from_mean_normalized).numpy() * vc_previous
    # background_gp_normalized = [] # clearing memory

    # Compute VAE(truth)
    latent_samples_true = tf.Variable(tf.random.normal(mean=true_mean, stddev=true_std, shape=(args.ensemble, autoencoder.latent_dim)))
    decoded_truth = autoencoder.decode(latent_samples_true)

    # And finally, lets decode the fitted latent space
    fitted = autoencoder.decode(latent)


    fig = Figure(
        figsize=(15, 60),
        dpi=100,
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
    cbars = 4
    if args.obs_increment != 0:
        cbars += 1
    tot_plots = 12
    height_sum = tot_plots * 0.075 + cbars * 0.015 + 0.005 * (tot_plots + cbars)
    dheight_plot = 0.075 / height_sum
    dheight_cbar = 0.015 / height_sum
    dheight_buffer = 0.005 / height_sum
    plot_bottom = 1

    lm = get_land_mask()
    # 1st (=top) - original field, obs. points
    plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
    ax_of = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_of.set_aspect("equal")
    ax_of.set_xticks([])
    ax_of.set_yticks([])
    ax_of.set_xlim(-180, 180)
    ax_of.set_ylim(-90, 90)
    if not fake_zeros:
        ax_of.set_ylabel('Truth for %04d-%02d-%02d' % (args.year, args.month, args.day) + r' [$\degree$C]')
    else:
        ax_of.set_ylabel('Climatology for %04d-%02d-%02d' % (args.year, args.month, args.day) + r' [$\degree$C]')
    ofp = plot_Earth(
        ax_of,
        tf.squeeze(t_in).numpy() * vc,
        vMin=-10,
        vMax=10,
        obs=tf.squeeze(t_obs, [0]).numpy(),
        o_size=0.5,
        land=lm,
        #label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
    )

    print(args.plot_singles, args.obs_increment)
    if args.plot_singles and args.obs_increment == 0:
        print('Im in')
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        pc = ax2.pcolormesh(lons, lats, tf.squeeze(t_in).numpy() * vc, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                            cmap='seismic')
        x = (tf.squeeze(t_obs, [0]).numpy()[:, 1] / 1440) * 360 - 180
        y = (tf.squeeze(t_obs, [0]).numpy()[:, 0] / 720) * 180 - 90
        y *= -1
        ax2.scatter(x, y, c='gold', s=5.0, marker='*', edgecolor='k',
                    linewidth=0.5, zorder=10 ** 4, transform=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.set_global()

        cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                      label=r'$\Delta T_{850}$ [$\degree$C]')

        fontsize = 16
        ticklabs = cb2.ax.get_yticklabels()
        cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
        ax2.set_title(r'Truth and obs. points', y=1.02, fontsize=fontsize)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        fig2.savefig(
            f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_truth.jpg',
            dpi=300)


    plot_bottom += dheight_plot + dheight_buffer
    ax_ocb = fig.add_axes([(1-0.5)/2, plot_bottom, 0.5, dheight_cbar])
    plot_colourbar(fig, ax_ocb, ofp)
    plot_bottom -= dheight_plot - dheight_buffer
    print('got 1st')

    # 2nd mean autoencoded original field
    plot_bottom -= dheight_plot + dheight_buffer
    dt_mean = tf.math.reduce_mean(decoded_truth, axis=0)

    ax_of2 = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_of2.set_aspect("equal")
    ax_of2.set_xticks([])
    ax_of2.set_yticks([])
    ax_of2.set_xlim(-180, 180)
    ax_of2.set_ylim(-90, 90)
    if not fake_zeros:
        ax_of2.set_ylabel(r'Mean autoencoded truth [$\degree$C]')
    else:
        ax_of2.set_ylabel(r'Mean autoencoded climatology [$\degree$C]')
    ofp = plot_Earth(
        ax_of2,
        tf.squeeze(dt_mean).numpy() * vc,
        vMin=-10,
        vMax=10,
        #obs=tf.squeeze(t_obs, [0]).numpy(),
        #o_size=0.5,
        land=lm,
        #label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
    )
    if args.plot_singles and args.obs_increment == 0:
        print('Im in')
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        pc = ax2.pcolormesh(lons, lats, tf.squeeze(dt_mean).numpy() * vc, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                            cmap='seismic')
        ax2.coastlines()
        ax2.set_global()

        cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                      label=r'$\Delta T_{850}$ [$\degree$C]')

        fontsize = 16
        ticklabs = cb2.ax.get_yticklabels()
        cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
        ax2.set_title(r'VAE(truth)', y=1.02, fontsize=fontsize)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        fig2.savefig(
            f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_VAE_of_turth.jpg',
            dpi=300)

    # 3rd decoded first guess
    plot_bottom -= dheight_plot + dheight_buffer
    ax_dfg = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_dfg.set_aspect("equal")
    ax_dfg.set_xticks([])
    ax_dfg.set_yticks([])
    ax_dfg.set_xlim(-180, 180)
    ax_dfg.set_ylim(-90, 90)
    dfgp = plot_Earth(
        ax_dfg,
        background_gp_mean,#        background_gp_from_mean,
        vMin=-10,
        vMax=10,
        land=lm,
    )
    ax_dfg.set_ylabel(r'D(background) [$\degree$C]')
    if args.plot_singles and args.obs_increment == 0:
        print('Im in')
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        pc = ax2.pcolormesh(lons, lats, background_gp_mean, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                            cmap='seismic')
        ax2.coastlines()
        ax2.set_global()

        cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                      label=r'$\Delta T_{850}$ [$\degree$C]')

        fontsize = 16
        ticklabs = cb2.ax.get_yticklabels()
        cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
        ax2.set_title(r'Background', y=1.02, fontsize=fontsize)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        fig2.savefig(
            f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_background.jpg',
            dpi=300)
    # 4th - mean output field = analysis
    plot_bottom -= dheight_plot + dheight_buffer
    e_mean = tf.math.reduce_mean(fitted, axis=0)
    e_std = tf.math.reduce_std(fitted, axis=0)

    ax_ef1 = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_ef1.set_aspect("equal")
    ax_ef1.set_xticks([])
    ax_ef1.set_yticks([])
    ax_ef1.set_xlim(-180, 180)
    ax_ef1.set_ylim(-90, 90)
    ax_ef1.set_ylabel(r"Analysis [$\degree$C]")
    plot_Earth(
        ax_ef1,
        tf.squeeze(e_mean).numpy() * vc,
        vMin=-10,
        vMax=10,
        land=lm,
        #label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
    )
    if args.plot_singles and args.obs_increment == 0:
        print('Im in')
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_mean).numpy() * vc, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                            cmap='seismic')
        ax2.coastlines()
        ax2.set_global()

        cb2 = fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                      label=r'$\Delta T_{850}$ [$\degree$C]')

        fontsize = 16
        ticklabs = cb2.ax.get_yticklabels()
        cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
        ax2.set_title(r'Analysis', y=1.02, fontsize=fontsize)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        fig2.savefig(
            f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_analysis.jpg',
            dpi=300)

    # 5th - analysis (=mean output field) - truth
    plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
    ax_amt = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_amt.set_aspect("equal")
    ax_amt.set_xticks([])
    ax_amt.set_yticks([])
    ax_amt.set_xlim(-180, 180)
    ax_amt.set_ylim(-90, 90)
    if not fake_zeros:
        ax_amt.set_ylabel(r"Analysis - truth [$\degree$C]")
    else:
        ax_amt.set_ylabel(r"Analysis - climatology [$\degree$C]")
    diff = tf.squeeze(e_mean).numpy() * vc - tf.squeeze(t_in).numpy() * vc
    minmax = max(abs(np.amin(diff)), abs(np.amax(diff)))
    efp = plot_Earth(
        ax_amt,
        diff,
        vMin=-10,
        vMax=10,
        #fog=tf.squeeze((e_std / c_std)).numpy(),
        #fog_threshold=0.33,
        land=lm,
        #label="Difference: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
    )
    plot_bottom += dheight_plot + dheight_buffer
    ax_amtcb = fig.add_axes([(1-0.5)/2, plot_bottom, 0.5, dheight_cbar])
    plot_colourbar(fig, ax_amtcb, efp)
    plot_bottom -= dheight_plot + dheight_buffer

    if args.obs_increment == 0.0 and args.plot_singles:
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        pc = ax2.pcolormesh(lons, lats, diff, transform=ccrs.PlateCarree(), vmin=-10,
                            vmax=10,
                            cmap='seismic')
        ax2.coastlines()
        ax2.set_global()

        cb2 = fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                            label=r'$T_{850}$ [$\degree$C]')
    
        fontsize = 16
        ticklabs = cb2.ax.get_yticklabels()
        cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
        ax2.set_title(r'Analysis - truth', y=1.02, fontsize=fontsize)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        fig2.savefig(
            f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_analysis_minus_truth.jpg',
            dpi=300)
        # print('diff', diff)
        # print('diff**2', diff**2)
        # print('type(lats)', type(lats))
        # print('np.cos(lats)', np.cos(lats))
        # print('diff**2*np.cos(lats)', diff**2*np.cos(lats))
        # print('np.mean(diff**2*np.cos(lats))', np.mean(diff**2*np.cos(lats)))
        # print('np.sqrt(np.mean(diff**2*np.cos(lats)))', np.sqrt(np.mean(diff**2*np.cos(lats))))



    lats = lm.coord("latitude").points
    lons = lm.coord("longitude").points
    lons, lats = np.meshgrid(lons, lats)
    rmse_analysis_gp = np.sqrt(np.mean((diff ** 2) * np.cos(np.radians(lats))) / np.mean(np.cos(np.radians(lats))))
    print('RMSE analysis gp', rmse_analysis_gp, '(only meaningful if obs. are sampled from the true field, not in case of preset observation increments)')

    # 6th - analysis (=mean output field) - decoded first guess
    vmin, vmax = -10, 10
    if args.obs_increment != 0.0:
        vmin, vmax = -5, 5
        plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
    else:
        vmin, vmax = -10, 10
        plot_bottom -= dheight_plot + dheight_buffer
    ax_amf = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_amf.set_aspect("equal")
    ax_amf.set_xticks([])
    ax_amf.set_yticks([])
    ax_amf.set_xlim(-180, 180)
    ax_amf.set_ylim(-90, 90)
    ax_amf.set_ylabel(r"Analysis increment [$\degree$C]")
    diff = tf.squeeze(e_mean).numpy() * vc - background_gp_mean #background_gp_from_mean #- tf.squeeze(t_previous_in).numpy() * vc_previous
    minmax = max(abs(np.amin(diff)), abs(np.amax(diff)))
    amf = plot_Earth(
        ax_amf,
        diff,
        vMin=vmin,
        vMax=vmax,
        #fog=tf.squeeze((e_std / c_std)).numpy(),
        #fog_threshold=0.33,
        land=lm,
        #label="Difference: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
    )
    if args.obs_increment != 0:
        plot_bottom += dheight_plot + dheight_buffer
        ax_amfcb = fig.add_axes([(1 - 0.5) / 2, plot_bottom, 0.5, dheight_cbar])
        plot_colourbar(fig, ax_amfcb, amf)
        plot_bottom -= dheight_plot + dheight_buffer

        print('First guess increment:', 0)
        print('Obs increment', np.mean(pseudo_obs) - tf.squeeze(interpolate_bilinear(tf.reshape(tf.convert_to_tensor(background_gp_mean, tf.float32), [1,720,1440,1]), t_obs, indexing="ij"), [0, 2]))
        print('Obs std', np.std(pseudo_obs))
        print('Analysis increment:', tf.squeeze(interpolate_bilinear(tf.reshape(tf.convert_to_tensor(diff, tf.float32), [1,720,1440,1]), t_obs, indexing="ij"), [0, 2]))

    if args.plot_singles:
        if args.obs_increment != 0:
            central_latitude = (1 - t_obs[0][0][0] / 720) * 180 - 90
            central_longitude = t_obs[0][0][1] / 1440 * 360 - 180
            if 'Singapore' in args.custom_addon:
                central_longitude_shift, central_latitude_shift = 30, 0
            else:
                central_longitude_shift, central_latitude_shift = 0, 0

            if abs(central_latitude) <= 15:
                vmin, vmax = -0.6, 0.6
                ticks = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            else:
                vmin, vmax = -3, 3
                ticks = [-3, -2, -1, 0, 1, 2, 3]

            fontsize = 16
            transform = ccrs.Orthographic(central_longitude=central_longitude + central_longitude_shift, central_latitude=central_latitude + central_latitude_shift)
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            print('entering pc')
            pc = ax2.pcolormesh(lons, lats, diff,
                                transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='seismic')
            color='gold'
            edgecolor='k' #'gold' #'k'
            size=80.0   #160.0
            linewidth=0.5 #1.0 #
            ax2.scatter([central_longitude], [central_latitude], c=color, s=size, marker='*', edgecolor=edgecolor, linewidth=linewidth, zorder=10**4, transform=ccrs.PlateCarree())
            print('adding coastlines')
            ax2.coastlines()
            print('setting global')
            ax2.set_global()
            cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, extend='both', label=r'$T_{850}$ [$\degree$C]', ticks=ticks)#, ticks=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]) #
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            #ax2.set_title(r"Analysis increment", y=1.02, fontsize=fontsize)
            if args.year != 2019 or args.month != 4 or args.day != 15:
                ax2.set_title('%04d-%02d-%02d' % (args.year, args.month, args.day), y=1.02, fontsize=fontsize)
            else:
                ax2.set_title('Analysis increment', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            print('saving')
            if not args.diagonal_B:
                fig2.savefig(
                    f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{args.custom_addon}_analysis_increment-non-diagonal_B.jpg', dpi=300
                    )
                raise AttributeError    # Not diagonal B-matrix
            else:
                fig2.savefig(
                    f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{args.custom_addon}_analysis_increment.jpg',
                    dpi=300
                )
                # if args.year != 2019 or args.month != 4 or args.day != 15:
                #     raise AttributeError # Not 2019-04-15
            print('saved single fig.')
            if color != 'gold' or edgecolor != 'k' or size != 80.0 or linewidth != 0.5:
                raise AttributeError

        else:
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            pc = ax2.pcolormesh(lons, lats, diff, transform=ccrs.PlateCarree(), vmin=-10,
                                vmax=10,
                                cmap='seismic')
            ax2.coastlines()
            ax2.set_global()

            cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                          label=r'$T_{850}$ [$\degree$C]')

            fontsize=16
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            ax2.set_title(r'Analysis increment', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_analysis_increment.jpg',
                dpi=300)

    # 7th truth - decoded 1st guess
    plot_bottom -= dheight_plot + dheight_buffer
    ax_tmf = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_tmf.set_aspect("equal")
    ax_tmf.set_xticks([])
    ax_tmf.set_yticks([])
    ax_tmf.set_xlim(-180, 180)
    ax_tmf.set_ylim(-90, 90)
    if not fake_zeros:
        ax_tmf.set_ylabel(r"Truth - D(background) [$\degree$C]")
    else:
        ax_tmf.set_ylabel(r"VAE(climatology) - D(background) [$\degree$C]")
    diff = tf.squeeze(t_in).numpy() * vc - background_gp_mean #background_gp_from_mean # - tf.squeeze(t_previous_in).numpy() * vc_previous
    minmax = max(abs(np.amin(diff)), abs(np.amax(diff)))
    efp = plot_Earth(
        ax_tmf,
        diff,
        vMin=-10,
        vMax=10,
        #fog=tf.squeeze((e_std / c_std)).numpy(),
        #fog_threshold=0.33,
        land=lm,
        #label="Difference: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
    )
    rmse_background_gp = np.sqrt(np.mean((diff**2 * np.cos(np.radians(lats)))) / np.mean(np.cos(np.radians(lats))))
    print('RMSE background gp', rmse_background_gp, '(only meaningful if obs. are sampled from the true field, not in case of preset observation increments)')

    # 8th - analysis - mean VAE(truth)
    plot_bottom -= dheight_plot + dheight_buffer
    ax_amvaet = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_amvaet.set_aspect("equal")
    ax_amvaet.set_xticks([])
    ax_amvaet.set_yticks([])
    ax_amvaet.set_xlim(-180, 180)
    ax_amvaet.set_ylim(-90, 90)
    ax_amvaet.set_ylabel(r"Analysis - mean VAE(truth) [$\degree$C]")
    diff = tf.squeeze(e_mean).numpy() * vc - tf.squeeze(dt_mean).numpy() * vc
    minmax = max(abs(np.amin(diff)), abs(np.amax(diff)))
    efp = plot_Earth(
        ax_amvaet,
        diff,
        vMin=-10,
        vMax=10,
        land=lm,
    )
    if args.plot_singles and args.obs_increment == 0:
        print('Im in')
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        pc = ax2.pcolormesh(lons, lats, diff, transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                            cmap='seismic')
        ax2.coastlines()
        ax2.set_global()

        cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                      label=r'$T_{850}$ [$\degree$C]')

        fontsize = 16
        ticklabs = cb2.ax.get_yticklabels()
        cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
        ax2.set_title(r'Analysis - VAE(truth)', y=1.02, fontsize=fontsize)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        fig2.savefig(
            f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_analysis_minus_VAE_of_truth.jpg',
            dpi=300)

        # Also background - VAE(truth)
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        lats = lm.coord("latitude").points
        lons = lm.coord("longitude").points
        lons, lats = np.meshgrid(lons, lats)
        pc = ax2.pcolormesh(lons, lats, background_gp_mean - tf.squeeze(dt_mean).numpy() * vc,
                            transform=ccrs.PlateCarree(), vmin=-10, vmax=10,
                            cmap='seismic')
        ax2.coastlines()
        ax2.set_global()

        cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='both',
                      label=r'$T_{850}$ [$\degree$C]')
        fontsize=16
        ticklabs = cb2.ax.get_yticklabels()
        cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
        ax2.set_title(r'Background - VAE(truth)', y=1.02, fontsize=fontsize)
        gl = ax2.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )
        fig2.savefig(
            f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_background_minus_VAE_of_truth.jpg',
            dpi=300)

    # 9th std of decoded background and of observations
    plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
    ax_std_ob = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_std_ob.set_aspect("equal")
    ax_std_ob.set_xticks([])
    ax_std_ob.set_yticks([])
    ax_std_ob.set_xlim(-180, 180)
    ax_std_ob.set_ylim(-90, 90)
    ax_std_ob.set_ylabel(r'Std of D(background) and of obs. [$\degree$C]')
    stdob = plot_Earth(
        ax_std_ob,
        background_gp_std,
        vMin=0,
        vMax=max(np.amax(background_gp_std), np.amax(obs_gp_std)),
        fog=None,#tf.squeeze((e_std / c_std)).numpy(),
        fog_threshold=0.1,
        land=lm,
        obs=tf.squeeze(t_obs, [0]).numpy(),
        o_size=3.0,
        obs_c=obs_gp_std,
        #label="Uncertainty: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
        cmap='terrain_r'
    )
    plot_bottom += dheight_plot + dheight_buffer
    ax_stdcb = fig.add_axes([(1-0.5)/2, plot_bottom, 0.5, dheight_cbar])
    plot_colourbar(fig, ax_stdcb, stdob)
    plot_bottom -= dheight_plot + dheight_buffer

    if args.plot_singles:
        if args.obs_increment != 0:
            central_latitude = (1 - t_obs[0][0][0] / 720) * 180 - 90
            central_longitude = t_obs[0][0][1] / 1440 * 360 - 180

            fontsize = 16
            transform = ccrs.Orthographic(central_longitude=central_longitude + central_longitude_shift,
                                          central_latitude=central_latitude + central_latitude_shift)
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            print('entering pc')
            pc = ax2.pcolormesh(lons, lats, background_gp_std,
                                transform=ccrs.PlateCarree(), vmin=0, vmax=5, cmap='terrain_r')
            ax2.scatter([central_longitude], [central_latitude], c='gold', s=80.0, marker='*', edgecolor='k', linewidth=0.5, zorder=10**4,
                                transform=ccrs.PlateCarree())
            print('adding coastlines')
            ax2.coastlines()
            print('setting global')
            ax2.set_global()
            cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, extend='max', label=r'$\sigma_b$ [$\degree$C]')
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            ax2.set_title(r'Std of background', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            print('saving')
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{args.custom_addon}_background_std.jpg', dpi=300
                )
            print('saved single fig.')

        else:
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            pc = ax2.pcolormesh(lons, lats, background_gp_std, transform=ccrs.PlateCarree(), vmin=0,
                                vmax=5,
                                cmap='terrain_r')
            ax2.coastlines()
            ax2.set_global()

            cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='max',
                          label=r'$\sigma_b$ [$\degree$C]')

            fontsize=16
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            ax2.set_title(r'Std of background', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_background_std.jpg',
                dpi=300)

    # 10th std of output field
    plot_bottom -= dheight_plot + dheight_buffer
    ax_std = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_std.set_aspect("equal")
    ax_std.set_xticks([])
    ax_std.set_yticks([])
    ax_std.set_xlim(-180, 180)
    ax_std.set_ylim(-90, 90)
    ax_std.set_ylabel(r'Std of analysis [$\degree$C]')
    stdp = plot_Earth(
        ax_std,
        tf.squeeze(e_std).numpy() * vc,
        vMin=0,
        vMax=max(np.amax(background_gp_std), np.amax(obs_gp_std)),
        fog=None,#tf.squeeze((e_std / c_std)).numpy(),
        fog_threshold=0.1,
        land=lm,
        obs=tf.squeeze(t_obs, [0]).numpy(),
        o_size=0.5,
        #label="Uncertainty: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
        cmap='terrain_r'
    )

    if args.obs_increment != 0:
        print('Background std:', tf.squeeze(interpolate_bilinear(tf.reshape(tf.convert_to_tensor(background_gp_std, tf.float32), [1,720,1440,1]), t_obs, indexing="ij"), [0, 2]))
        print('Analysis std:', tf.squeeze(interpolate_bilinear(tf.reshape(tf.convert_to_tensor(tf.squeeze(e_std).numpy() * vc, tf.float32), [1,720,1440,1]), t_obs, indexing="ij"), [0, 2]))

    if args.plot_singles:
        if args.obs_increment != 0:
            central_latitude = (1 - t_obs[0][0][0] / 720) * 180 - 90
            central_longitude = t_obs[0][0][1] / 1440 * 360 - 180

            fontsize = 16
            transform = ccrs.Orthographic(central_longitude=central_longitude + central_longitude_shift,
                                          central_latitude=central_latitude + central_latitude_shift)
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            print('entering pc')
            pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_std).numpy() * vc,
                                transform=ccrs.PlateCarree(), vmin=0, vmax=5, cmap='terrain_r')
            # ax2.scatter([central_longitude], [central_latitude], c=obs_gp_std, edgecolor='k', linewidth=2,
            #             s=15.0, marker='o', zorder=10**4, vmin=0, vmax=5, cmap='terrain_r')
            print('adding coastlines')
            ax2.coastlines()
            ax2.scatter([central_longitude], [central_latitude], c='gold', s=80.0, marker='*', edgecolor='k',
                        linewidth=0.5, zorder=10 ** 4, transform=ccrs.PlateCarree())
            print('setting global')
            ax2.set_global()
            cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, extend='max', label=r'$\sigma_a$ [$\degree$C]')
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            ax2.set_title(r'Std of analysis', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            print('saving')
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{args.custom_addon}_analysis_std.jpg', dpi=300
                )
            print('saved single fig.')
        else:
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_std).numpy() * vc,
                                transform=ccrs.PlateCarree(), vmin=0, vmax=5, cmap='terrain_r')
            ax2.coastlines()
            ax2.set_global()

            cb2=fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='max',
                          label=r'$\sigma_a$ [$\degree$C]')
            fontsize=16
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            ax2.set_title(r'Std of analysis', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_analysis_std.jpg',
                dpi=300)


    # 11th ratio between std of analysis and std of background
    if args.obs_increment != 0:
        vminr, vmaxr = 0.5, 1.5
    else:
        vminr, vmaxr = 0, 2
    plot_bottom -= dheight_plot + dheight_buffer + dheight_cbar + dheight_buffer
    ax_std_ratio = fig.add_axes([plot_left, plot_bottom, plot_width, dheight_plot])
    ax_std_ratio.set_aspect("equal")
    ax_std_ratio.set_xticks([])
    ax_std_ratio.set_yticks([])
    ax_std_ratio.set_xlim(-180, 180)
    ax_std_ratio.set_ylim(-90, 90)
    ax_std_ratio.set_ylabel(r'Std of analysis / std of background')
    stdp_ratio = plot_Earth(
        ax_std_ratio,
        tf.squeeze(e_std).numpy() * vc / background_gp_std,
        vMin=vminr,
        vMax=vmaxr,
        fog=None,  # tf.squeeze((e_std / c_std)).numpy(),
        fog_threshold=0.1,
        land=lm,
        obs=tf.squeeze(t_obs, [0]).numpy(),
        o_size=0.5,
        # label="Uncertainty: %04d-%02d-%02d" % (args.oyear, args.month, args.day),
        cmap='PiYG_r'
    )
    plot_bottom += dheight_plot + dheight_buffer
    ax_std_ratio_cbar = fig.add_axes([(1 - 0.5) / 2, plot_bottom, 0.5, dheight_cbar])
    plot_colourbar(fig, ax_std_ratio_cbar, stdp_ratio)
    plot_bottom -= dheight_plot + dheight_buffer

    if args.plot_singles:
        if args.obs_increment != 0:
            central_latitude = (1 - t_obs[0][0][0] / 720) * 180 - 90
            central_longitude = t_obs[0][0][1] / 1440 * 360 - 180

            if abs(central_latitude) < 5:
                vmin, vmax = 0.9, 1.1
                ticks = [0.90, 0.95, 1, 1.05, 1.10]
            elif 'CAR' in args.custom_addon:
                vmin, vmax = 0.8, 1.2
                ticks = [0.8, 0.9, 1, 1.1, 1.2]
            else:
                vmin, vmax = 0.50, 1.50
                ticks = [0.50, 0.75, 1, 1.25, 1.50]

            fontsize = 16
            transform = ccrs.Orthographic(central_longitude=central_longitude + central_longitude_shift,
                                          central_latitude=central_latitude + central_latitude_shift)
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            print('entering pc')
            pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_std).numpy() * vc / background_gp_std,
                                transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='PiYG_r')
            ax2.scatter([central_longitude], [central_latitude], c='gold', s=80.0, marker='*', edgecolor='k',
                        transform=ccrs.PlateCarree(),
                        linewidth=0.5, zorder=10 ** 4)
            print('adding coastlines')
            ax2.coastlines()
            print('setting global')
            ax2.set_global()
            cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, extend='both', ticks=ticks, label=r'$\sigma_{a} / \sigma_{b}$')
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            ax2.set_title(r'Std ratio', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            print('saving')
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{args.custom_addon}_std_ratio.jpg', dpi=300
                )
            print('saved single fig.')
        else:
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            lats = lm.coord("latitude").points
            lons = lm.coord("longitude").points
            lons, lats = np.meshgrid(lons, lats)
            pc = ax2.pcolormesh(lons, lats, tf.squeeze(e_std).numpy() * vc / background_gp_std,
                                transform=ccrs.PlateCarree(), vmin=0, vmax=2, cmap='PiYG_r')
            ax2.coastlines()
            ax2.set_global()

            cb2 = fig2.colorbar(pc, ax=ax2, location='bottom', shrink=0.8, pad=0.05, extend='max', label=r'$\sigma_{a} / \sigma_{b}$')
            fontsize=16
            ticklabs = cb2.ax.get_yticklabels()
            cb2.ax.set_yticklabels(ticklabs, fontsize=fontsize)
            ax2.set_title(r'Std ratio', y=1.02, fontsize=fontsize)
            gl = ax2.gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_std_ratio.jpg',
                dpi=300)


    # 12th box chart of latent variables
    plot_bottom -= dheight_plot + dheight_buffer
    width_ef = 0.5  # Tired of Googling and ChatGPT
    ax_box = fig.add_axes([(1 - width_ef)/2, plot_bottom, width_ef, dheight_plot*0.95])
    print(np.shape(previous_latents.numpy()))   #.numpy
    # previous_latents_theoretical = np.array([scipy.stats.norm.ppf(i, loc=previous_true_mean, scale=np.sqrt(np.diagonal(B_matrix))) for i in np.linspace(0.005,0.995,100)])
    # bpltfirstguess = ax_box.boxplot(previous_latents, whis=[5,95], showfliers=False)
    bpltfirstguess = ax_box.boxplot(previous_latents.numpy(), whis=[5, 95], showfliers=False)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bpltfirstguess[element], color='C3')
    if args.obs_increment == 0.0:
        latent_teoretical = np.array([scipy.stats.norm.ppf(i, loc=true_mean, scale=true_std) for i in np.linspace(0.005,0.995,100)])
        print(np.shape(latent_teoretical), np.shape(lnpy))#
        #input('printed')
        bpltrue = ax_box.boxplot(latent_teoretical, whis=[5,95], showfliers=False)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bpltrue[element], color='C1')
    bpl = ax_box.boxplot(lnpy, whis=[5,95], flierprops={'marker': 'o', 'markersize': 1, 'markerfacecolor': 'C0', 'markeredgecolor': 'C0'})
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bpl[element], color='C0')
    ax_box.set_xticks([i for i in range(1, autoencoder.latent_dim+1) if (i-1)%5==0], [i for i in range(autoencoder.latent_dim) if i%5==0])
    ax_box.grid(linestyle=':', linewidth=0.6, color='gray')
    ax_box.set_ylabel('Latent element value')
    ax_box.set_xlabel('Index in latent space')
    if args.obs_increment == 0.0:
        ax_box.legend([bpltrue["boxes"][0], bpltfirstguess["boxes"][0], bpl["boxes"][0]], ['Encoded truth', 'Background', 'Analysis'], loc='upper right', framealpha=0.6)
    else:
        ax_box.legend([bpltfirstguess["boxes"][0], bpl["boxes"][0]],
                      ['Background', 'Analysis'], loc='upper right', framealpha=0.6)

    if args.plot_singles:

        fontsize = 16
        fig2 = plt.figure(figsize=(10, 4))
        ax_box = fig2.add_subplot(1, 1, 1)
        linewidths = {'boxes': 0.5, 'whiskers': 0.1}

        bpltfirstguess = ax_box.boxplot(previous_latents.numpy(), whis=[5, 95], showfliers=False)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element == 'medians':
                plt.setp(bpltfirstguess[element], color='C3', alpha=1, solid_capstyle="butt")
            elif element not in linewidths:
                plt.setp(bpltfirstguess[element], color='C3', alpha=1)
            elif element == 'whiskers':
                plt.setp(bpltfirstguess[element], color='C3', alpha=1, linewidth=linewidths[element], linestyle=':')
            else:
                plt.setp(bpltfirstguess[element], color='C3', alpha=1, linewidth=linewidths[element])
        if args.obs_increment == 0.0:
            latent_teoretical = np.array(
                [scipy.stats.norm.ppf(i, loc=true_mean, scale=true_std) for i in np.linspace(0.005, 0.995, 100)])
            print(np.shape(latent_teoretical), np.shape(lnpy))  #
            # input('printed')
            bpltrue = ax_box.boxplot(latent_teoretical, whis=[5, 95], showfliers=False)
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bpltrue[element], color='C1')
        bpl = ax_box.boxplot(lnpy, whis=[5, 95], showfliers=False)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element == 'medians':
                plt.setp(bpl[element], color='C0', alpha=0.8, solid_capstyle="butt")
            elif element not in linewidths:
                plt.setp(bpl[element], color='C0', alpha=0.8)
            elif element == 'whiskers':
                plt.setp(bpl[element], color='C0', alpha=0.8, linewidth=linewidths[element], linestyle=':')
            else:
                plt.setp(bpl[element], color='C0', alpha=0.8, linewidth=linewidths[element])
        ax_box.set_xticks([i for i in range(1, autoencoder.latent_dim + 1) if (i - 1) % 5 == 0],
                          [i for i in range(autoencoder.latent_dim) if i % 5 == 0])
        ax_box.grid(linestyle=':', linewidth=0.6, color='gray')
        ax_box.set_ylabel('Latent element value')
        ax_box.set_xlabel('Index in latent space')
        if args.obs_increment == 0.0:
            ax_box.legend([bpltrue["boxes"][0], bpltfirstguess["boxes"][0], bpl["boxes"][0]],
                          ['Encoded truth', 'Background', 'Analysis'], loc='upper right', framealpha=0.6)
        else:
            ax_box.legend([bpltfirstguess["boxes"][0], bpl["boxes"][0]],
                          ['Background', 'Analysis'], loc='upper right', framealpha=0.6)

        fig2.tight_layout()



        print('saving')
        if args.obs_increment == 0.0:
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{args.custom_addon}_latent.pdf'
                )
            print('saved single fig.')
        else:
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_latent.jpg',
                dpi=300)


    mean_background_latent = np.mean(previous_latents.numpy(), axis=0)
    mean_analysis_latent = np.mean(latent, axis=0)
    std_background_latent = np.std(previous_latents.numpy(), axis=0)
    std_analysis_latent = np.std(latent, axis=0)
    if args.obs_increment == 0.0:
        mean_true_latent = np.mean(latent_teoretical, axis=0)
        rmse_background_latent = np.mean((mean_true_latent - mean_background_latent)**2)
        rmse_analysis_latent = np.mean((mean_analysis_latent - mean_true_latent)**2)
        print('rmse_background_latent', rmse_background_latent)
        print('rmse_analysis_latent', rmse_analysis_latent)
    print('number of std reductions:', sum([1 if std_analysis_latent[ilatelem] < std_background_latent[ilatelem] else 0 for ilatelem in range(len(std_analysis_latent))]))
    print('number of std increases:', sum([0 if std_analysis_latent[ilatelem] < std_background_latent[ilatelem] else 1 for ilatelem in range(len(std_analysis_latent))]))
    print('average analysis latent std', np.mean(np.std(latent, axis=0)))
    print('average background latent std', np.mean(np.std(previous_latents.numpy(), axis=0)))
    # ax_box.text(.85, .15, f'RMSE(median) = {rmse_background:.2f}', color='C3', transform=plt.gca().transAxes)
    # ax_box.text(.85, .05, f'RMSE(median) = {rmse_analysis:.2f}', color='C0', transform=plt.gca().transAxes)

    # for ie in range(len(true_mean)):
    #     ax_box.plot([ie+1, ie+1, ie+1], [true_mean[ie] - true_half_iqr[ie], true_mean[ie], true_mean[ie] + true_half_iqr[ie]], color='C1', linewidth=0.5, zorder=1, marker='_', markersize=6)
    #     print([ie+1, ie+1, ie+1], [true_mean[ie] - true_half_iqr[ie], true_mean[ie], true_mean[ie] + true_half_iqr[ie]])
    #ax_box.set_xlim(-1, autoencoder.latent_dim)
    #ax_box.set_ylim(np.amin(lnpy), np.amax(lnpy))

    # 12th loss and logpz during minimization
    # ax_minimization = fig.add_axes([(1 - width_ef) / 2, 0.01, width_ef, 0.035])
    # intermediate_logpzs = the_3DVar_outputs[3]
    # intermediate_losses = the_3DVar_outputs[2]
    # steps = [i for i in range(len(intermediate_logpzs))]
    # ax_minimization.axhline(log_normal_pdf(true_mean, 0., 0., raxis=0)/autoencoder.latent_dim, color='C1')
    # ax_minimization.axhline(log_normal_pdf(previous_true_mean, 0., 0., raxis=0)/autoencoder.latent_dim, color='C2')
    # ax_minimization.axhline(-1.4125933495324293, color='k')
    # ax_minimization.plot(steps, intermediate_logpzs, color='C0')
    # ax_minimization.grid(linestyle=':', linewidth=0.6, color='gray')
    # ax_minimization.set_xlabel('Minimization step')
    # ax_minimization.set_ylabel('logpz (solid line)')
    # ax_minimization.axvline(np.argmin(intermediate_losses), linestyle='-.', color='gray')
    # ax_minimization.set_xlim(0, max(steps))
    # ax_minimization1 = ax_minimization.twinx()
    # ax_minimization1.plot(steps, intermediate_losses, linestyle='--', color='C0')
    # ax_minimization1.set_ylabel('loss (dashed line)')
    if args.plot_singles:
        idx = 1
        print('len(loss)', len(loss[idx]))
        print('loss ratios', [loss[idx][i]/loss[idx][i-1] for i in range(1, len(loss[idx]))])
        fontsize = 16
        fig2 = plt.figure(figsize=(12, 4))
        ax2 = fig2.add_subplot(1, 4, 1)
        steps = [i for i in range(len(loss[idx]))]
        ax2.plot(loss[idx], label=r'$\mathcal{J}_z$')
        ax2.legend()
        ax2.set_xlabel('Minimisation step')
        ax2.set_ylabel('Total cost')
        ax2.set_xlim(0, max(steps))
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        ax3 = fig2.add_subplot(1, 4, 2, sharex=ax2)
        ax3.plot(all_Jo[idx], label=r'$\mathcal{J}_{oz}$')
        ax3.legend()
        ax3.set_xlabel('Minimisation step')
        ax3.set_ylabel('Observation term')
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        print('Jo ratios', [all_Jo[idx][i]/all_Jo[idx][i-1] for i in range(1, len(loss[idx]))])

        ax4 = fig2.add_subplot(1, 4, 3, sharex=ax2)
        ax4.plot(np.array(loss[idx]) -np.array(all_Jo[idx]), label=r'$\mathcal{J}_{bz}$')
        ax4.legend()
        ax4.set_xlabel('Minimisation step')
        ax4.set_ylabel('Background term')
        ax4.set_ylim(bottom=0)
        ax4.set_yticks([0, 50, 100])    # Set manually for better appearance
        ax4.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        print('Jb ratios', [(np.array(loss[idx]) -np.array(all_Jo[idx]))[i]/(np.array(loss[idx]) -np.array(all_Jo[idx]))[i-1] for i in range(1, len(loss[idx]))])

        ax5 = fig2.add_subplot(1, 4, 4, sharex=ax2)
        ax5.plot(all_gradient[idx], label=r'$||\nabla \mathcal{J}_{z}||$')
        ax5.legend()
        ax5.set_xlabel('Minimisation step')
        ax5.set_ylabel('Gradient')
        ax5.set_ylim(bottom=0)
        ax5.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # ax3 = fig2.add_subplot(2, 1, 2)
        # steps = [i for i in range(len(loss[0]))]
        # ax3.plot(steps, all_gradient[0], label=r'$||\nabla J_z||$')
        # ax3.legend()
        # ax3.set_xlabel('Minimization step')
        # ax3.set_ylabel('Value')

        fig2.tight_layout()

        if args.obs_increment == 0.0:
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_{args.custom_addon}_J{idx}.pdf'
            )
            print('saved single fig.')
        else:
            fig2.savefig(
                f'fit_multi-experiment005--Ensemble-3D-Var-figures/single_figs/fe005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_res={res}_{args.custom_addon}_J{idx}.jpg',
                dpi=300)





    # savefig_name = f'fit_multi-experiment004--3D-Var_obs_on_regular_grid-figures/fe003--Fit_pseudo_obs_on_quasi_regular_grid-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_ensemble={args.ensemble}_obs_std={args.obs_std}_res={res}'
    savefig_name = f'fit_multi-experiment005--Ensemble-3D-Var-figures/fe005--fit_multi-experiment005--Ensemble-3D-Var-{args.year}-{args.month}-{args.day}_epoch={args.epoch}_obs_std={args.obs_std}_res={res_str}_ensemble={args.ensemble}'
    if args.std_first_multiplier != 1.0:
        savefig_name += f'std_first_mutiplier={args.std_first_multiplier}'
    if args.minimization_learning_rate != 0.01:
        savefig_name += f'minimization_lr={args.minimization_learning_rate}'
    if not args.adaptive_lr:
        savefig_name += '_no_adaptive_lr'
    if args.perfect_first:
        savefig_name += '_perfect_first'
    if args.perfect_obs:
        savefig_name += '_perfect_obs'
    if args.diagonal_B:
        savefig_name += '_diagonal_B'
    if args.flow_B_fraction > 0:
        savefig_name += f'_flow_B_fraction={args.flow_B_fraction}'
        if args.ERA5_ens_mem_spread_multiplicator != 1:
            savefig_name += f'_ERA5_ens_spr_mul={args.ERA5_ens_mem_spread_multiplicator}'
    if args.cycling_starting_point != '':
        savefig_name += f'_cycling_from_{args.cycling_starting_point}'
    if len(args.custom_addon) > 0:
        savefig_name += '_' + args.custom_addon

    if args.cycling_starting_point != '':
        # Save RMSE and don't save the figure
        rmse_filename = ('fit_multi-experiment005--Ensemble-3D-Var-data/' + ''.join(savefig_name.split('/')[1:]))
        rmse_filename = ''.join(rmse_filename.split(f'{args.year}-{args.month}-{args.day}_')) + '.pkl'
        rmse_filename = ''.join(rmse_filename.split('--fit_multi-experiment005'))
        print(rmse_filename)
        raise AssertionError
        try:
            rmse_dict = pickle.load(open(rmse_filename, 'rb'))
        except: # File does not yet exist
            rmse_dict = {}
        rmse_dict[f'{args.year}-{args.month:02d}-{args.day:02d}'] = {}
        rmse_dict[f'{args.year}-{args.month:02d}-{args.day:02d}']['rmse_background_gp'] = rmse_background_gp
        rmse_dict[f'{args.year}-{args.month:02d}-{args.day:02d}']['rmse_analysis_gp'] = rmse_analysis_gp
        rmse_dict[f'{args.year}-{args.month:02d}-{args.day:02d}']['rmse_background_latent'] = rmse_background_latent
        rmse_dict[f'{args.year}-{args.month:02d}-{args.day:02d}']['rmse_analysis_latent'] = rmse_analysis_latent

        pickle.dump(rmse_dict, open(rmse_filename, 'wb'))

        raise AttributeError    # Not saving the figure for cycling experiments!

    fig.savefig(savefig_name + ".jpg", dpi=300, facecolor="white")
    finish = datetime.datetime.now()
    print(str(finish - start))
    print('saved jpg')

    if args.save_as_pdf:
        fig.savefig(savefig_name + ".pdf", facecolor="white")
        finish = datetime.datetime.now()
        print(str(finish - start))
        print('saved pdf')