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
parser.add_argument("--compute_differences", help='Go through all dates and compute differences in latent space for consecutive dates', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--compute_B_matrix", help='Compute B matrix from differences in latent space for consecutive dates', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--plot", help='Plot B matrix', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--parallel_processes", help="Number of CPUs used fo computing differences", type=int, default=4, required=False)
parser.add_argument("--start_year", type=int, required=False, default=2015)
parser.add_argument("--start_month", type=int, required=False, default=1)
parser.add_argument("--start_day", type=int, required=False, default=1)
parser.add_argument("--end_year", type=int, required=False, default=2018)
parser.add_argument("--end_month", type=int, required=False, default=12)
parser.add_argument("--end_day", type=int, required=False, default=31)
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

# If computing the differences, set these to dates of interest.
# If only computing the B-matrix, set these in a way that their pkl file already exists and
# that the dates for B-matrix are a subset of these dates.
start_date_differences = date(args.start_year, args.start_month, args.start_day) #date(2015, 1, 1)
end_date_differences = date(args.end_year, args.end_month, args.end_day)#date(2018, 12, 31)

start_date_B_matrix = date(args.start_year, args.start_month, args.start_day)#date(2015, 1, 1)
end_date_B_matrix = date(args.end_year, args.end_month, args.end_day)#date(2018, 12, 31)

delta_differences = end_date_differences - start_date_differences   # returns timedelta
delta_B_matrix = end_date_B_matrix - start_date_B_matrix


if args.compute_differences:
    def change_in_latent_space(date):
        # Set up the model and load the weights at the chosen epoch
        sys.path.append("%s/.." % os.path.dirname(__file__))
        from autoencoderModel import DCVAE

        latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % (1000*(args.epoch//1000))]
        autoencoder = DCVAE(latent_dim=latent_dim)

        weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
            args.epoch,
        )
        # print('Uploading weights from', weights_dir)
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
        # Check the load worked
        devn = load_status.assert_existing_objects_matched()

        # We are using it in inference mode
        # (I'm not at all sure this actually works)
        autoencoder.decoder.trainable = False
        for layer in autoencoder.decoder.layers:
            layer.trainable = False
        autoencoder.decoder.compile()


        print(date)
        previous_t = ERA5_load_T850((date - timedelta(days=1)).year, (date - timedelta(days=1)).month,
                                   (date - timedelta(days=1)).day)
        c = ERA5_load_T850_climatology((date - timedelta(days=1)).year, (date - timedelta(days=1)).month,
                                      (date - timedelta(days=1)).day)
        vc = ERA5_load_T850_variability_climatology((date - timedelta(days=1)).year,
                                                   (date - timedelta(days=1)).month,
                                                   (date - timedelta(days=1)).day)
        previous_t = previous_t - c
        previous_t = previous_t / vc
        previous_t = ERA5_roll_longitude(previous_t)
        previous_t_in = tf.convert_to_tensor(previous_t.data, np.float32)
        previous_t_in = tf.reshape(previous_t_in, [1, 720, 1440, 1])
        previous_latent_mean, previous_latent_logvar = autoencoder.encode(previous_t_in)
        plm = previous_latent_mean.numpy().astype('float16')

        new_t = ERA5_load_T850(date.year, date.month, date.day)
        new_c = ERA5_load_T850_climatology(date.year, date.month, date.day)
        new_vc = ERA5_load_T850_variability_climatology(date.year, date.month, date.day)
        new_t = new_t - new_c
        new_t = new_t / new_vc
        new_t = ERA5_roll_longitude(new_t)
        new_t_in = tf.convert_to_tensor(new_t.data, np.float32)
        new_t_in = tf.reshape(new_t_in, [1, 720, 1440, 1])
        new_latent_mean, new_latent_logvar = autoencoder.encode(new_t_in)
        nlm = new_latent_mean.numpy().astype('float16')

        difference_mean = nlm - plm
        difference_mean = np.reshape(difference_mean, (autoencoder.latent_dim))
        error_matrix = np.tensordot(difference_mean, difference_mean, axes=0).astype('float16')   # tensor product
        print('error_matrix.dtype', error_matrix.dtype)

        del autoencoder
        gc.collect()

        return {date:error_matrix}

    if __name__ == '__main__':
        inputs = [start_date_differences + timedelta(days=d) for d in range(delta_differences.days + 1)]
        with multiprocessing.Pool(processes=args.parallel_processes) as pool:
            # Use map() to parallelize the execution of the tasks
            results = pool.map(change_in_latent_space, inputs)

        output_matrices = results[0].copy()
        for result in results[1:]:
            output_matrices.update(result)

        print('Dumping output_matrices')
        if args.epoch == 1020:
            pickle.dump(output_matrices, open('decoding_experiment006---B-matrix_for_persistence-data/error_matrices_' + start_date_differences.strftime('%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'wb'))
        else:
            pickle.dump(output_matrices, open(
                f'decoding_experiment006---B-matrix_for_persistence-data/error_matrices_epoch={args.epoch}_' + start_date_differences.strftime(
                    '%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'wb'))

        print('Dumped output_matrices')

if args.compute_B_matrix:
    if start_date_B_matrix < date(2015, 1, 1) and end_date_B_matrix > date(2014, 12, 31) and args.epoch==1020:
        dict_of_matrices = pickle.load(open('decoding_experiment006---B-matrix_for_persistence-data/error_matrices_1979-01-02_to_2014-12-31.pkl', 'rb'))
        dict_of_matrices1 = pickle.load(open('decoding_experiment006---B-matrix_for_persistence-data/error_matrices_2015-01-01_to_2022-12-31.pkl', 'rb'))
        dict_of_matrices.update(dict_of_matrices1)
    else:
        if args.epoch==1020:
            dict_of_matrices = pickle.load(open('decoding_experiment006---B-matrix_for_persistence-data/error_matrices_' + start_date_differences.strftime('%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'rb'))
        else:
            dict_of_matrices = pickle.load(open(
                f'decoding_experiment006---B-matrix_for_persistence-data/error_matrices_epoch={args.epoch}_' + start_date_differences.strftime(
                    '%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'rb'))
    #print(len(dict_of_matrices.keys()))
    #print(delta_differences.days)
    matrices_of_interest = np.array([dict_of_matrices[start_date_B_matrix + timedelta(days=d)] for d in range(delta_B_matrix.days + 1)])
    #print(np.shape(matrices_of_interest))
    B_matrix = np.mean(matrices_of_interest, axis=0)
    #print(np.shape(B_matrix))
    #print('diag')
    #print(np.diagonal(B_matrix))
    #print('first line')
    #print(B_matrix[0,:])
    #print(np.amin(np.diagonal(B_matrix)))
    #print(np.amax(np.abs((B_matrix - B_matrix*np.identity(100)))))

    if args.epoch == 1020:
        pickle.dump(B_matrix, open(f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_' + start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pkl', 'wb'))
    else:
        pickle.dump([B_matrix, len(matrices_of_interest)], open(
            f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_epoch={args.epoch}_' + start_date_B_matrix.strftime(
                '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pkl', 'wb'))

if args.plot:
    latent_dim = pickle.load(open('../models_by_epochs/latent_dims.pkl', 'rb'))['%04d' % (1000 * (args.epoch // 1000))]
    if args.epoch != 1020:
        B_matrix = pickle.load(open(
            f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_epoch={args.epoch}_' + start_date_B_matrix.strftime(
                '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pkl', 'rb'))
    else:
        B_matrix = pickle.load(open(f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_' + start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pkl', 'rb'))

    if len(B_matrix) == 2:
        B_matrix = B_matrix[0]

    from matplotlib.colors import LogNorm

    plt.figure(figsize=(12, 12))
    #matplotlib.rcParams.update({"font.size": 12})
    mat = plt.matshow(np.abs(B_matrix), norm=LogNorm(vmin=1e-4, vmax=1), cmap='gist_earth_r')
    plt.colorbar(mat, fraction=0.092, pad=0.03, shrink=0.85, extend='both')
    plt.title(r'abs($\mathbf{B}_z$), included dates ' + start_date_B_matrix.strftime('%Y-%m-%d') + ' to ' + end_date_B_matrix.strftime('%Y-%m-%d'))
    #plt.tight_layout()
    if args.epoch == 1020:
        plt.savefig('decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-' + start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.jpg', dpi=300)
        plt.savefig('decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-' + start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pdf', dpi=300)
    else:
        plt.savefig(
            f'decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-epoch={args.epoch}_' + start_date_B_matrix.strftime(
                '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.jpg', dpi=300)
        plt.savefig(
            f'decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-epoch={args.epoch}_' + start_date_B_matrix.strftime(
                '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pdf', dpi=300)
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

    nbin = 21
    bins = np.linspace(-4,0, nbin)
    plt.hist(np.log10(np.diagonal(B_matrix)), bins=bins, density=True, alpha=0.8, label='Diagonal elements')
    plt.hist(np.log10(np.abs(B_matrix - B_matrix * np.identity(latent_dim)).flatten()), bins=bins, density=True, alpha=0.8, label='Off-diagonal elements')
    plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z$ element))')
    plt.xlim(min(bins), max(bins))
    plt.ylabel('Percentage')
    plt.legend()
    plt.title(r'Distribution of $\mathbf{B}_z$ elements')
    plt.yticks(ticks=nbin/(max(bins) - min(bins)) * np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), labels=['0', '20', '40', '60', '80', '100']) # to mapiranje postudiraj
    plt.tight_layout()
    plt.savefig('decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-' +
                start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') +
                'hist.pdf', dpi=300)
