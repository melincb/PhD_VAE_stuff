#!/usr/bin/env python


import os
import sys
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

import random
from scipy.stats import pearsonr

import iris
import IRData.twcr as twcr
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

autoencoder = DCVAE()

weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
    args.epoch,
)
#print('Uploading weights from', weights_dir)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
# Check the load worked
devn = load_status.assert_existing_objects_matched()

# We are using it in inference mode
# (I'm not at all sure this actually works)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()


start_date = date(2019, 1, 1) 
end_date = date(2022, 12, 31)

delta = end_date - start_date   # returns timedelta


previous_t = ERA5_load_T850((start_date - timedelta(days=1)).year, (start_date - timedelta(days=1)).month, (start_date - timedelta(days=1)).day)
c = ERA5_load_T850_climatology((start_date - timedelta(days=1)).year, (start_date - timedelta(days=1)).month, (start_date - timedelta(days=1)).day)
vc = ERA5_load_T850_variability_climatology((start_date - timedelta(days=1)).year, (start_date - timedelta(days=1)).month, (start_date - timedelta(days=1)).day)
previous_t = previous_t - c
previous_t = previous_t / vc
previous_t = ERA5_roll_longitude(previous_t)
previous_t_in = tf.convert_to_tensor(previous_t.data, np.float32)
previous_t_in = tf.reshape(previous_t_in, [1, 720, 1440, 1])
previous_latent_mean, previous_latent_logvar = autoencoder.encode(previous_t_in)
plm = previous_latent_mean.numpy()
vc = vc.data

difference_mean = [[] for i in range(autoencoder.latent_dim)]
difference_to_random_mean = [[] for i in range(autoencoder.latent_dim)]
starting_mean = [[] for i in range(autoencoder.latent_dim)]

for i in range(delta.days + 1): #1.1.2019-31.12.2022
    day = start_date + timedelta(days=i)
    new_t = ERA5_load_T850(day.year, day.month, day.day)
    new_c = ERA5_load_T850_climatology(day.year, day.month, day.day)
    new_vc = ERA5_load_T850_variability_climatology(day.year, day.month, day.day)
    new_t = new_t - new_c
    new_t = new_t / new_vc
    new_t = ERA5_roll_longitude(new_t)
    new_t_in = tf.convert_to_tensor(new_t.data, np.float32)
    new_t_in = tf.reshape(new_t_in, [1, 720, 1440, 1])
    print(day)
    new_latent_mean, new_latent_logvar = autoencoder.encode(new_t_in)
    random_start = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim))).numpy()
    nlm = new_latent_mean.numpy()
    #print(nlm[0,:5])
    for j in range(len(nlm[0])):
        difference_mean[j].append(nlm[0,j] - plm[0,j])
        starting_mean[j].append(plm[0,j])
        difference_to_random_mean[j].append(nlm[0,j] - random_start[0,j])
    #print(np.shape(difference_mean))
    #previous_latent_mean, previous_latent_logvar = new_latent_mean.copy(), new_latent_logvar.copy()
    plm = nlm.copy()

difference_mean, starting_mean, difference_to_random_mean = np.array(difference_mean), np.array(starting_mean), np.array(difference_to_random_mean)
print(np.shape(difference_mean))
print(np.shape(np.abs(difference_mean)))
abs_difference_mean = np.abs(difference_mean)
mean_abs_difference_mean = np.mean(abs_difference_mean, axis=1)
print('shape mean_abs_difference_mean', np.shape(mean_abs_difference_mean))
std_abs_difference_mean = np.std(abs_difference_mean, axis=1)
abs_starting_mean = np.abs(starting_mean)
print(np.shape(abs_starting_mean))
print(np.shape(abs_difference_mean))
print(type(abs_difference_mean))
abs_difference_to_random_mean = np.abs(difference_to_random_mean)

lat_elem = [ie for ie in range(autoencoder.latent_dim)]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5*10/7,9))

#plt.errorbar(lat_elem, mean_abs_difference_mean, yerr=std_abs_difference_mean, fmt='o')
if random:
    bplasm = ax1.boxplot(np.transpose(abs_difference_to_random_mean), whis=[5,95], showfliers=False) # 
else:
    bplasm = ax1.boxplot(np.transpose(abs_starting_mean), whis=[5,95], showfliers=False) # , label='abs(mean of "today")'
linewidths = {'boxes':0.5, 'whiskers':0.1}#, 'means':3}
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    if element == 'medians':
        plt.setp(bplasm[element], color='C1', alpha=1, solid_capstyle="butt")
    elif element not in linewidths:
        plt.setp(bplasm[element], color='C1', alpha=1)
    elif element=='whiskers':
        plt.setp(bplasm[element], color='C1', alpha=1, linewidth=linewidths[element], linestyle=':')
    else:
        plt.setp(bplasm[element], color='C1', alpha=1, linewidth=linewidths[element])
bplmadm = ax1.boxplot(np.transpose(abs_difference_mean), whis=[5,95], showfliers=False) # , label='abs(diff. between mean of "yesterday" and "today")'
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    if element == 'medians':
        plt.setp(bplmadm[element], color='C0', alpha=0.6, solid_capstyle="butt")
    elif element not in linewidths:
        plt.setp(bplmadm[element], color='C0', alpha=0.6)
    elif element=='whiskers':
        plt.setp(bplmadm[element], color='C0', alpha=0.6, linewidth=linewidths[element], linestyle=':')
    else:
        plt.setp(bplmadm[element], color='C0', alpha=0.6, linewidth=linewidths[element])
ax1.set_xticks([i for i in range(1, autoencoder.latent_dim+1) if (i-1)%5==0], [i for i in range(autoencoder.latent_dim) if i%5==0])
ax1.set_ylim(bottom=0)
ax1.set_xlabel('Latent element index')
ax1.set_ylabel('Distribution')
if random:
    'abs(mean of day of interest - mean of previous day)'
    'abs(mean of day of interest - random sample)'
    ax1.legend([bplmadm["boxes"][0], bplasm["boxes"][0]], ['abs(mean of day of interest - mean of previous day)', 'abs(mean of day of interest - random sample)'], loc='upper right', framealpha=0.6)
else:
    ax1.legend([bplmadm["boxes"][0], bplasm["boxes"][0]], ['abs(diff. between mean of "yesterday" and "today")', 'abs(mean of "today")'], loc='upper right', framealpha=0.6)

ax2.scatter(np.array(lat_elem) + 1, [pearsonr(abs_starting_mean[ie], abs_difference_mean[ie])[0] for ie in range(autoencoder.latent_dim)], c='k', label='R(abs(previous day mean), abs(diff. of means))')
ax2.set_xlabel('Latent element index')
ax2.set_ylabel('Correlation coefficient')
ax2.set_xticks([i for i in range(1, autoencoder.latent_dim+1) if (i-1)%5==0], [i for i in range(autoencoder.latent_dim) if i%5==0])
xll, xlr = ax1.get_xlim()
ax2.set_xlim(xll, xlr)
ax2.legend()

plt.suptitle('Changes in latent space for time step of 1 day\nafter feeding the encoder the "truth"')

plt.tight_layout()
if random:
    plt.savefig(f'decoding_experiment005--Changes_in_latent_space_after_encoding_truth-figures/de005--Changes_in_latent_space_after_encoding_truth-timestep=1day_epoch={args.epoch}_dates=' + start_date.strftime("%Y-%m-%d") + '_to_' + end_date.strftime("%Y-%m-%d") + '_+random.jpg', dpi=300)
    print('saved jpg')
    plt.savefig(f'decoding_experiment005--Changes_in_latent_space_after_encoding_truth-figures/de005--Changes_in_latent_space_after_encoding_truth-timestep=1day_epoch={args.epoch}_dates=' + start_date.strftime("%Y-%m-%d") + '_to_' + end_date.strftime("%Y-%m-%d") + '_+random.pdf')
    print('saved pdf')
else:
    plt.savefig(f'decoding_experiment005--Changes_in_latent_space_after_encoding_truth-figures/de005--Changes_in_latent_space_after_encoding_truth-timestep=1day_epoch={args.epoch}_dates=' + start_date.strftime("%Y-%m-%d") + '_to_' + end_date.strftime("%Y-%m-%d") + '.jpg', dpi=300)
    print('saved jpg')
    plt.savefig(f'decoding_experiment005--Changes_in_latent_space_after_encoding_truth-figures/de005--Changes_in_latent_space_after_encoding_truth-timestep=1day_epoch={args.epoch}_dates=' + start_date.strftime("%Y-%m-%d") + '_to_' + end_date.strftime("%Y-%m-%d") + '.pdf')
    print('saved pdf')