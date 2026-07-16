#!/usr/bin/env python

import multiprocessing

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

import matplotlib.pyplot as plt

import argparse
import pickle

import time
import gc


def parallel_worker(input):
    date = input['date']
    ensemble = input['ensemble']
    training_type = input['training_type']
    epoch_range = input['epoch_range']
    constant = input['constant']

    output = compute_for_this_date(date=date,
                                   ensemble=ensemble,
                                   training_type=training_type,
                                   epoch_range=epoch_range,
                                   constant=constant)
    return output

def compute_for_this_date(date, ensemble, training_type, epoch_range, constant):
    sys.path.append("%s/../validation" % os.path.dirname(__file__))
    from plot_ERA5_comparison import get_land_mask  # The real point here is to get the path to ERA5_load
    from ERA5_load import ERA5_load_T850_variability_climatology
    from ERA5_load import ERA5_roll_longitude
    vc = ERA5_load_T850_variability_climatology(1981, date.month, date.day)
    vc = ERA5_roll_longitude(vc)
    vc = vc.data
    vc = vc.astype(np.float32)

    sys.path.append("%s/.." % os.path.dirname(__file__))
    from autoencoderModel import DCVAE

    weighted_means, weighted_successes = [], []

    phi = np.array([[lat*np.pi/180 for lon in range(1440)] for lat in np.linspace(-89.875, 89.875, 720, endpoint=True)])
    cosphi = np.cos(phi)
    cosphi = cosphi.astype(np.float32)
    weighted_successful_max = np.sum(cosphi)
    latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % training_type]
    autoencoder = DCVAE(latent_dim=latent_dim)

    for epoch in range(training_type + 1, training_type + epoch_range + 1):  # training_type + 1
        print(date, epoch)


        # Set up the model and load the weights at the chosen epoch
        #autoencoder = DCVAE()
        weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
            epoch,
        )
        # print('Uploading weights from', weights_dir)
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
        # Check the load worked
        devn = load_status.assert_existing_objects_matched()
        # del autoencoder
        # time.sleep(2)

        # We are using it in inference mode
        # (I'm not at all sure this actually works)
        autoencoder.decoder.trainable = False
        for layer in autoencoder.decoder.layers:
            layer.trainable = False
        autoencoder.decoder.compile()
        #time.sleep(2)

        # latent = tf.Variable(constant * np.ones(shape=(1, autoencoder.latent_dim)))    # Bostjan: For constant values of all members of latent space
        latent = tf.Variable(tf.random.normal(stddev=constant, shape=(
                                        ensemble, autoencoder.latent_dim)))  # Bostjan: for random values of members of latent space
        fitted = tf.cast(autoencoder.decode(latent), dtype=tf.float32)
        e_mean = tf.math.reduce_mean(fitted, axis=0)
        e_std = tf.math.reduce_std(fitted, axis=0)
        del fitted
        e_mean = tf.squeeze(e_mean).numpy() * vc
        e_std = tf.squeeze(e_std).numpy() * vc
        # print(np.shape(e_mean.numpy()))
        e_mean_plus_std = e_mean + e_std
        e_mean_minus_std = e_mean - e_std
        e_times = e_mean_minus_std * e_mean_plus_std  # Bostjan + * - = -, if successful

        weighted_successful = np.sum(np.where(e_times <= 0, 1, 0) * cosphi)

        weighted_successes.append(weighted_successful / weighted_successful_max * 100)
        weighted_means.append(np.mean(e_mean * cosphi) / np.mean(cosphi))

        # print('Sizes for epoch:', epoch,
        #       '\n e_mean', e_mean.nbytes,
        #       '\n e_std', e_std.nbytes,
        #       '\n e_mean_plus_std', e_mean_plus_std.nbytes,
        #       '\n e_mean_minus_std', e_mean_minus_std.nbytes,
        #       '\n e_times', e_times.nbytes,
        #       '\n weighted means', np.array(weighted_means).nbytes,
        #       '\n weighted successes', np.array(weighted_successes).nbytes)
        # print('ws', weighted_successes)
        # print('wm', weighted_means)

        # del e_mean
        # del e_std
        # del e_mean_plus_std
        # del e_mean_minus_std
        # del e_times
        # del autoencoder

        gc.collect()

    return {'weighted_means':weighted_means, 'weighted_successes':weighted_successes}



if __name__ == '__main__':
    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_type', type=int, required=True)     #0000, 1000, 2000, 3000, 4000, 5000, 6000, 7000
    parser.add_argument("--ensemble", help="Ensemble size", type=int, required=False, default=5)
    parser.add_argument('--compute', help="Compute or only plot", default=True, action=argparse.BooleanOptionalAction)  #In order to only plot: --no-compute
    args = parser.parse_args()

    training_type = args.training_type

    epoch_ranges = {0000:100, 1000:100, 2000:100, 3000:100, 4000:100, 5000:84, 6000:84, 7000:67}
    epoch_range = epoch_ranges[training_type]

    constant = 1.0  # Standard deviation in normal random sampling

    if args.compute:
        dates = [datetime.datetime(year=1981, month=1, day=15),
                 datetime.datetime(year=1981, month=4, day=15),
                 datetime.datetime(year=1981, month=7, day=15),
                 datetime.datetime(year=1981, month=10, day=15)]

        inputs = [{'date':dates[idate],
                   'ensemble':args.ensemble,
                   'training_type':training_type,
                   'epoch_range':epoch_range,
                   'constant':constant} for idate in range(len(dates))]
        pool = multiprocessing.Pool(processes=4)
        results = pool.map(parallel_worker, inputs)
        pool.close()
        pool.join()
        weighted_means = [results[idate]['weighted_means'] for idate in range(len(dates))]
        weighted_successes = [results[idate]['weighted_successes'] for idate in range(len(dates))]
        pickle.dump({'weighted_means':weighted_means, 'weighted_successes':weighted_successes, 'dates':dates}, open(f"decoding_experiment002--Mean_world_temperature-figures/de002--Mean_world_temperature-mu=0_sigma={constant}_ensemble={args.ensemble}_training_type={training_type:04d}.pkl", 'wb'))
    else:
        means_successes = pickle.load(open(f"decoding_experiment002--Mean_world_temperature-figures/de002--Mean_world_temperature-mu=0_sigma={constant}_ensemble={args.ensemble}_training_type={training_type:04d}.pkl", 'rb'))
        weighted_means = means_successes['weighted_means']
        weighted_successes = means_successes['weighted_successes']
        dates = means_successes['dates']

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()
    c1 = 'r'
    c2 = 'b'
    linestyles = ['-', '--', '-.', ':']
    for idate in range(len(dates)):
        ax1.plot([ep for ep in range(1, len(weighted_means[idate])+1)], weighted_means[idate], color='k', linestyle=linestyles[idate], label=dates[idate].strftime('%d %b'))  # Just for the legend
        ax1.plot([ep for ep in range(1, len(weighted_means[idate])+1)], weighted_means[idate], color=c1, linestyle=linestyles[idate])
        ax2.plot([ep for ep in range(1, len(weighted_means[idate])+1)], weighted_successes[idate], color=c2, linestyle=linestyles[idate])

    max_epoch = np.amax([len(wm) for wm in weighted_means])

    ax1.set_ylabel(r'Mean world T850 anomaly [$\degree$C]', color=c1)
    ax1.set_xlabel('Epoch')
    ax1.tick_params(axis='y', labelcolor=c1)
    ax1.axhline(0, color='tab:red', linestyle='--', linewidth=1)
    ax1.grid(axis='x', color='grey', linestyle=':', linewidth=0.6)
    ax1.set_xlim(0, max_epoch)
    ax1.set_xticks([e for e in range(max_epoch) if e%10==0])
    ax1.legend(loc='lower right')

    ax2.set_ylabel(r'Success rate [$\%$]', color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    ax2.set_ylim(0,100)
    #ax2.set_yscale('log')
    deterministic_multiplier = pickle.load(open('../models_by_epochs/deterministic_multipliers.pkl', 'rb'))['%04d' % training_type]
    latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % training_type]
    plt.title('Huber norm multiplier: %.0e, latent space dimension: %d' % (deterministic_multiplier, latent_dim))

    plt.tight_layout()
    plt.savefig(f"decoding_experiment002--Mean_world_temperature-figures/de002--Mean-world_temperature-mu=0_sigma={constant}_ensemble={args.ensemble}_training_type={training_type:04d}.pdf", facecolor="white")
    finish = datetime.datetime.now()
    print(str(finish - start))