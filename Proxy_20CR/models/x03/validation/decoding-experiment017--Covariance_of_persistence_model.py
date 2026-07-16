#!/usr/bin/env python


import os
import sys
import numpy as np
import multiprocessing

print('impoerting tf')
import tensorflow as tf
print('impoerted tf')
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
parser.add_argument("--gridpoints", help="Grid points in each direction. 80 means central location +- 10 degNS +- 10 degEW", type=int, required=False, default=80)
parser.add_argument('--central_lat', help="Central latitude", type=float, required=True)
parser.add_argument('--central_lon', help="Central longitude", type=float, required=True)
parser.add_argument("--compute_covariances", help='Compute error variances and covariances over the desired date span', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--plot", help='Compute R-matrix and plot its line corresponding to correlation coeffitients for the central location', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--parallel_processes", help="Number of CPUs used fo computing differences", type=int, default=4, required=False)
parser.add_argument("--start_year", type=int, required=False, default=2015)
parser.add_argument("--start_month", type=int, required=False, default=1)
parser.add_argument("--start_day", type=int, required=False, default=1)
parser.add_argument("--end_year", type=int, required=False, default=2018)
parser.add_argument("--end_month", type=int, required=False, default=12)
parser.add_argument("--end_day", type=int, required=False, default=31)
parser.add_argument('--custom_addon', type=str, default='', required=False)
args = parser.parse_args()

# Functions for plotting
sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_ERA5_comparison import get_land_mask
from plot_ERA5_comparison import plot_Earth
from plot_ERA5_comparison import plot_colourbar


from ERA5_load import ERA5_load_T850
# from ERA5_load import ERA5_load_T850_climatology
# from ERA5_load import ERA5_load_T850_variability_climatology
from ERA5_load import ERA5_roll_longitude

import gc

# If computing the differences, set these to dates of interest.
# If only computing the B-matrix, set these in a way that their pkl file already exists and
# that the dates for B-matrix are a subset of these dates.
start_date_differences = date(args.start_year, args.start_month, args.start_day) #date(2015, 1, 1)
end_date_differences = date(args.end_year, args.end_month, args.end_day)#date(2018, 12, 31)


delta_differences = end_date_differences - start_date_differences   # returns timedelta

lats = np.array([[args.central_lat]])
lons = np.array([[args.central_lon]])
t_lats_npy = (lats.flatten() + 90) / 180  # orig. (obs["Latitude"].values + 90) / 180
t_lons_npy = (lons.flatten() + 180) / 360  # orig. (obs["Longitude"].values) / 360
t_lats = tf.convert_to_tensor(1.0 - t_lats_npy,
                              tf.float32)
t_lons = tf.convert_to_tensor(t_lons_npy, tf.float32)
t_obs = tf.stack((t_lats * 720, t_lons * 1440), axis=1)
t_obs = tf.expand_dims(t_obs, 0)

print('t_obs', t_obs)

# previous_t = ERA5_load_T850(2000, 1, 1)
# print('imported')
# previous_t = ERA5_roll_longitude(previous_t)
# print('rolled')
# previous_t = tf.convert_to_tensor(previous_t.data, np.float16)
# print('converted')
# previous_t = tf.reshape(previous_t, [1, 720, 1440, 1])
# print('reshaped')
# print('t_obs', t_obs)
# bil = interpolate_bilinear(previous_t, t_obs, indexing="ij")
# print('bil')
# squeeze = tf.squeeze(bil, [0, 2])
# print('squeeze')
# squeeze_npy = squeeze.numpy()
# print('squeezed npy')
# previous_exact = squeeze_npy#tf.squeeze(interpolate_bilinear(previous_t, t_obs, indexing="ij"), [0, 2]).numpy()
# print('squeezed')
# previous_t_zoom = previous_t[0,
#                   int(t_obs[0,0,0]) - args.gridpoints//2:int(t_obs[0,0,0]) + args.gridpoints//2,
#                   int(t_obs[0,0,1]) - args.gridpoints//2:int(t_obs[0,0,1]) + args.gridpoints//2,
#                   0].numpy()
# print('zoom')
# previous_total = np.concatenate((previous_exact, previous_t_zoom.flatten()))
# print('previous total')


if args.compute_covariances:
    def change_in_latent_space(date):
        print(date)


        previous_t = ERA5_load_T850((date - timedelta(days=1)).year, (date - timedelta(days=1)).month,
                                   (date - timedelta(days=1)).day)
        previous_t = ERA5_roll_longitude(previous_t)
        previous_t = tf.convert_to_tensor(previous_t.data, np.float16)
        previous_t = tf.reshape(previous_t, [1, 720, 1440, 1])



        bil = interpolate_bilinear(previous_t, t_obs, indexing="ij")
        squeeze = tf.squeeze(bil, [0, 2])
        squeeze_npy = squeeze.numpy()
        previous_exact = squeeze_npy#tf.squeeze(interpolate_bilinear(previous_t, t_obs, indexing="ij"), [0, 2]).numpy()
        previous_t_zoom = previous_t[0,
                          int(t_obs[0,0,0]) - args.gridpoints//2:int(t_obs[0,0,0]) + args.gridpoints//2,
                          int(t_obs[0,0,1]) - args.gridpoints//2:int(t_obs[0,0,1]) + args.gridpoints//2,
                          0].numpy()

        previous_total = np.concatenate((previous_exact, previous_t_zoom.flatten())).astype(np.float16)
        del previous_t
        gc.collect()

        new_t = ERA5_load_T850(date.year, date.month, date.day)
        new_t = ERA5_roll_longitude(new_t)
        new_t = tf.convert_to_tensor(new_t.data, np.float16)
        new_t = tf.reshape(new_t, [1, 720, 1440, 1])
        new_exact = tf.squeeze(interpolate_bilinear(new_t, t_obs, indexing="ij"), [0, 2]).numpy()
        new_t_zoom = new_t[0,
                          int(t_obs[0, 0, 0]) - args.gridpoints // 2:int(t_obs[0, 0, 0]) + args.gridpoints // 2,
                          int(t_obs[0, 0, 1]) - args.gridpoints // 2:int(t_obs[0, 0, 1]) + args.gridpoints // 2,
                          0].numpy()
        new_total = np.concatenate((new_exact, new_t_zoom.flatten())).astype(np.float16)

        difference = previous_total - new_total

        del new_t
        gc.collect()

        return difference

    if __name__ == '__main__':
        inputs = [start_date_differences + timedelta(days=d) for d in range(delta_differences.days + 1)]
        # with multiprocessing.Pool(processes=args.parallel_processes) as pool: # Nekaj ni vredu zaradi bilinearne interpolacije
        #     # Use map() to parallelize the execution of the tasks
        #     results = pool.map(change_in_latent_space, inputs)
        output_differences = []
        for date in inputs:
            output_differences.append(change_in_latent_space(date))

        #output_differences_transposed = output_differences.T
        print('Output differences shape', np.shape(output_differences))
        print('Dumping output_matrices')

        pickle.dump(output_differences, open(
            f'decoding_experiment017--Covariance_of_persistence_model-data/errors_{args.custom_addon}_{args.gridpoints}gp_'
            + start_date_differences.strftime('%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'wb'))

        print('Dumped output_matrices')

# if args.compute_B_matrix:
#     if start_date_B_matrix < date(2015, 1, 1) and end_date_B_matrix > date(2014, 12, 31) and args.epoch==1020:
#         dict_of_matrices = pickle.load(open('decoding_experiment006---B-matrix_for_persistence-data/error_matrices_1979-01-02_to_2014-12-31.pkl', 'rb'))
#         dict_of_matrices1 = pickle.load(open('decoding_experiment006---B-matrix_for_persistence-data/error_matrices_2015-01-01_to_2022-12-31.pkl', 'rb'))
#         dict_of_matrices.update(dict_of_matrices1)
#     else:
#         if args.epoch==1020:
#             dict_of_matrices = pickle.load(open('decoding_experiment006---B-matrix_for_persistence-data/error_matrices_' + start_date_differences.strftime('%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'rb'))
#         else:
#             dict_of_matrices = pickle.load(open(
#                 f'decoding_experiment006---B-matrix_for_persistence-data/error_matrices_epoch={args.epoch}_' + start_date_differences.strftime(
#                     '%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'rb'))
#     #print(len(dict_of_matrices.keys()))
#     #print(delta_differences.days)
#     matrices_of_interest = np.array([dict_of_matrices[start_date_B_matrix + timedelta(days=d)] for d in range(delta_B_matrix.days + 1)])
#     #print(np.shape(matrices_of_interest))
#     B_matrix = np.mean(matrices_of_interest, axis=0)
#     #print(np.shape(B_matrix))
#     #print('diag')
#     #print(np.diagonal(B_matrix))
#     #print('first line')
#     #print(B_matrix[0,:])
#     #print(np.amin(np.diagonal(B_matrix)))
#     #print(np.amax(np.abs((B_matrix - B_matrix*np.identity(100)))))
#
#     if args.epoch == 1020:
#         pickle.dump(B_matrix, open(f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_' + start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pkl', 'wb'))
#     else:
#         pickle.dump([B_matrix, len(matrices_of_interest)], open(
#             f'decoding_experiment006---B-matrix_for_persistence-data/B_matrix_epoch={args.epoch}_' + start_date_B_matrix.strftime(
#                 '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pkl', 'wb'))
#
if args.plot:
    output_differences = pickle.load(open(
            f'decoding_experiment017--Covariance_of_persistence_model-data/errors_{args.custom_addon}_{args.gridpoints}gp_'
            + start_date_differences.strftime('%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d') + '.pkl', 'rb'))

    print('output_differences', np.shape(output_differences))
    covariances = np.cov(np.array(output_differences).T)
    print('covariances', np.shape(covariances))
    variances = np.diagonal(covariances)
    print('variances', variances)
    covariances_with_central_point = covariances[0]
    correlation_coefficients = np.zeros((720, 1440))
    covariances2D = np.zeros((720, 1440))
    for_contour = np.zeros((720, 1440))
    min_ilat = int(t_obs[0, 0, 0]) - args.gridpoints // 2
    min_ilon = int(t_obs[0, 0, 1]) - args.gridpoints // 2
    for ilat in range(int(t_obs[0, 0, 0]) - args.gridpoints // 2, int(t_obs[0, 0, 0]) + args.gridpoints // 2):
        for ilon in range(int(t_obs[0, 0, 1]) - args.gridpoints // 2, int(t_obs[0, 0, 1]) + args.gridpoints // 2):
            idx = (ilat - min_ilat) * args.gridpoints + (ilon - min_ilon) + 1
            #print(idx)
            correlation_coefficients[ilat, ilon] = covariances_with_central_point[idx] / (np.sqrt(variances[0] * variances[idx]))
            covariances2D[ilat, ilon] = covariances_with_central_point[idx]
            for_contour[ilat, ilon] = 1

    import cartopy.crs as ccrs
    fig2 = plt.figure(figsize=(6, 4))
    lm = get_land_mask()
    transform = ccrs.Orthographic(central_longitude=args.central_lon,
                                  central_latitude=args.central_lat)
    ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
    lats = lm.coord("latitude").points
    #print('lats', lats)
    lons = lm.coord("longitude").points
    #print('lons', lons)
    lons, lats = np.meshgrid(lons, lats)
    pc = ax2.pcolormesh(lons, lats, correlation_coefficients, transform=ccrs.PlateCarree(), vmin=-1, vmax=1,
                        cmap='bwr')
    color = 'gold'
    edgecolor = 'k'  # 'gold' #'k'
    size = 80.0  # 160.0
    linewidth = 0.5  # 1.0 #
    ax2.scatter([args.central_lon], [args.central_lat], c=color, s=size, marker='*', edgecolor=edgecolor,
                linewidth=linewidth, zorder=10 ** 4, transform=ccrs.PlateCarree())
    ax2.contour(lons, lats, for_contour, levels=1, transform=ccrs.PlateCarree(), linewidths=1.5, colors='k', zorder=10)
    ax2.coastlines()
    ax2.set_global()
    cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, label=r'Background error correlation coefficient')
    ax2.set_title('Background error correlation coefficient\n with respect to observation location', y=1.02)
    gl = ax2.gridlines(
        draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    fig2.savefig(
        f"decoding_experiment017--Covariance_of_persistence_model-figures/single_figs/de017--Covariance_of_persistence_model_{args.custom_addon}_{args.gridpoints}gp_"
        + start_date_differences.strftime('%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d')
        + "R.jpg", dpi=300)


    fig2 = plt.figure(figsize=(6, 4))
    lm = get_land_mask()
    transform = ccrs.Orthographic(central_longitude=args.central_lon,
                                  central_latitude=args.central_lat)
    ax2 = fig2.add_subplot(1, 1, 1, projection=transform)
    lats = lm.coord("latitude").points
    #print('lats', lats)
    lons = lm.coord("longitude").points
    #print('lons', lons)
    lons, lats = np.meshgrid(lons, lats)
    pc = ax2.pcolormesh(lons, lats, covariances2D, transform=ccrs.PlateCarree(), vmin=-1, vmax=1,
                        cmap='bwr')
    color = 'gold'
    edgecolor = 'k'  # 'gold' #'k'
    size = 80.0  # 160.0
    linewidth = 0.5  # 1.0 #
    ax2.scatter([args.central_lon], [args.central_lat], c=color, s=size, marker='*', edgecolor=edgecolor,
                linewidth=linewidth, zorder=10 ** 4, transform=ccrs.PlateCarree())
    ax2.contour(lons, lats, for_contour, levels=1, transform=ccrs.PlateCarree(), linewidths=1.5, colors='k', zorder=10)
    ax2.coastlines()
    ax2.set_global()
    cb2 = fig2.colorbar(pc, ax=ax2, location='right', shrink=0.8, pad=0.05, extend='both', label=r'Background error corvariance [$\degree\mathrm{C}^2$]')
    ax2.set_title('Background error covariance\n with respect to observation location', y=1.02)
    gl = ax2.gridlines(
        draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    fig2.savefig(
        f"decoding_experiment017--Covariance_of_persistence_model-figures/single_figs/de017--Covariance_of_persistence_model_{args.custom_addon}_{args.gridpoints}gp_"
        + start_date_differences.strftime('%Y-%m-%d') + '_to_' + end_date_differences.strftime('%Y-%m-%d')
        + "cov.jpg", dpi=300)


#     if len(B_matrix) == 2:
#         B_matrix = B_matrix[0]
#
#     from matplotlib.colors import LogNorm
#
#     plt.figure(figsize=(12, 12))
#     #matplotlib.rcParams.update({"font.size": 12})
#     mat = plt.matshow(np.abs(B_matrix), norm=LogNorm(vmin=1e-4, vmax=1), cmap='gist_earth_r')
#     plt.colorbar(mat, fraction=0.092, pad=0.03, shrink=0.85, extend='both')
#     plt.title(r'abs($\mathbf{B}_z$), included dates ' + start_date_B_matrix.strftime('%Y-%m-%d') + ' to ' + end_date_B_matrix.strftime('%Y-%m-%d'))
#     #plt.tight_layout()
#     if args.epoch == 1020:
#         plt.savefig('decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-' + start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.jpg', dpi=300)
#         plt.savefig('decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-' + start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pdf', dpi=300)
#     else:
#         plt.savefig(
#             f'decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-epoch={args.epoch}_' + start_date_B_matrix.strftime(
#                 '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.jpg', dpi=300)
#         plt.savefig(
#             f'decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-epoch={args.epoch}_' + start_date_B_matrix.strftime(
#                 '%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') + '.pdf', dpi=300)
#     print('done')
#
#     print('min diag. element', np.amin(np.diagonal(B_matrix)))
#     print('max diag. element', np.amax(np.diagonal(B_matrix)))
#     print('max offdiag. element', np.amax(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
#     print('mean diag. element', np.mean(np.diagonal(B_matrix)))
#     print('mean offdiag. element', np.mean(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
#     print('ratio of means',
#           np.mean(np.diagonal(B_matrix)) / np.mean(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
#     print('sum diag. elements / sum offdiag. elements',
#           np.sum(np.diagonal(B_matrix)) / np.sum(np.abs((B_matrix - B_matrix * np.identity(latent_dim)))))
#     max_offdiag_values = np.max(np.abs((B_matrix - B_matrix * np.identity(latent_dim))), axis=1)
#     print('number of elements with larger offdiag. than diag. value',
#           np.sum(np.where(np.diagonal(B_matrix) < max_offdiag_values, 1, 0)))
#     print('worst ratio between diag. and offdiag. value', np.max(max_offdiag_values / np.diagonal(B_matrix)))
#
#     plt.cla()
#     plt.clf()
#     plt.figure(figsize=(4*1.05*1.017, 4*1.05*1.017))
#
#     nbin = 21
#     bins = np.linspace(-4,0, nbin)
#     plt.hist(np.log10(np.diagonal(B_matrix)), bins=bins, density=True, alpha=0.8, label='Diagonal elements')
#     plt.hist(np.log10(np.abs(B_matrix - B_matrix * np.identity(latent_dim)).flatten()), bins=bins, density=True, alpha=0.8, label='Off-diagonal elements')
#     plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z$ element))')
#     plt.xlim(min(bins), max(bins))
#     plt.ylabel('Percentage')
#     plt.legend()
#     plt.title(r'Distribution of $\mathbf{B}_z$ elements')
#     plt.yticks(ticks=nbin/(max(bins) - min(bins)) * np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), labels=['0', '20', '40', '60', '80', '100']) # to mapiranje postudiraj
#     plt.tight_layout()
#     plt.savefig('decoding_experiment006---B-matrix_for_persistence-figures/de006--B-matrix_for_persistence-' +
#                 start_date_B_matrix.strftime('%Y-%m-%d') + '_to_' + end_date_B_matrix.strftime('%Y-%m-%d') +
#                 'hist.pdf', dpi=300)
