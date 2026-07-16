#!/usr/bin/env python

# This program computes PSD of the output from decoding-experiment010--Spectrum_of_VAE_of_truth-pyshtools.py.
# It needs to be run in pyshtools virtual environment.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import argparse
import pickle

from pyshtools.expand import SHExpandDH

parser = argparse.ArgumentParser()
parser.add_argument("--compute", help='Compute PSD or just plot it', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=1020)
parser.add_argument("--ensemble", help="Ensemble size", type=int, required=False, default=150)
parser.add_argument("--year", type=int, required=False, default=2019)
parser.add_argument("--month", type=int, required=False, default=4)
parser.add_argument("--day", type=int, required=False, default=15)
args = parser.parse_args()



if args.compute:
    fields = pickle.load(open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch={args.epoch:04d}_ensemble={args.ensemble}_decoded_fields.pkl', 'rb'))
    truth = False
    # Uncomment the following 2 lines to compute the spectrum of the truth
    # fields = np.reshape(pickle.load(open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_TRUTH_field.pkl', 'rb')), (1,720,1440))
    # truth = True

    def compute_power_spectra(field):
        field = np.array(field)

        coeffs_field = SHExpandDH(field, sampling=2)
        #coeffs_field_mod = SHExpandDH(field, sampling=1)

        # print(coeffs_field)
        # print(len(coeffs_field), len(coeffs_field[0]), len(coeffs_field[0][0]))

        coeff_amp = coeffs_field[0, :, :] ** 2 + coeffs_field[1, :, :] ** 2
        #coeff_amp_mod = coeffs_field_mod[0, :, :] ** 2 + coeffs_field_mod[1, :, :] ** 2

        spectra_zon = np.sum(coeff_amp, axis=0)
        spectra_mer = np.sum(coeff_amp, axis=1)

        return spectra_zon, spectra_mer

    psds_zon = []
    psds_mer = []
    for field in fields:
        psd_zon, psd_mer = compute_power_spectra(field)
        psds_zon.append(psd_zon)
        psds_mer.append(psd_mer)
    print('np.shape(psds_zon)', np.shape(psds_zon))
    print('np.shape(np.array(psds_zon))', np.shape(np.array(psds_zon)))
    if truth:
        pickle.dump([np.array(psds_zon), np.array(psds_mer)], open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_TRUTH_field_psds.pkl', 'wb'))
    else:
        pickle.dump([np.array(psds_zon), np.array(psds_mer)], open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch={args.epoch:04d}_ensemble={args.ensemble}_psds.pkl', 'wb'))
    # Clearing space on disk (overrunning the pkl file with fields (150 fields -> approx 800 MB)
    pickle.dump([], open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch={args.epoch:04d}_ensemble={args.ensemble}_decoded_fields.pkl', 'wb'))


#plt.figure(figsize=(12,4))
plt.subplot(1,2,1)  # THERE IS STH WRONG WITH THE "ZONAL" SPECTRUM
ld100_zon, ld100_mer = pickle.load(open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch=1020_ensemble={args.ensemble}_psds.pkl', 'rb'))
print(np.shape(ld100_zon))
print(np.min(ld100_zon, axis=0)[10], np.max(ld100_zon, axis=0)[10])
plt.plot([i for i in range(len(ld100_zon[0]))], np.mean(ld100_zon, axis=0), label=r'$N=100$')

ld50_zon, ld50_mer = pickle.load(open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch=6040_ensemble={args.ensemble}_psds.pkl', 'rb'))
print(np.shape(ld50_zon))
print(np.min(ld50_zon, axis=0)[10], np.max(ld50_zon, axis=0)[10])
plt.plot([i for i in range(len(ld50_zon[0]))], np.mean(ld50_zon, axis=0), label=r'$N=50$')

ld200_zon, ld200_mer = pickle.load(open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch=5070_ensemble={args.ensemble}_psds.pkl', 'rb'))
print(np.shape(ld200_zon))
print(np.min(ld200_zon, axis=0)[10], np.max(ld200_zon, axis=0)[10])
plt.plot([i for i in range(len(ld200_zon[0]))], np.mean(ld200_zon, axis=0), label=r'$N=200$')

ld2000_zon, ld2000_mer = pickle.load(open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_epoch=7040_ensemble={args.ensemble}_psds.pkl', 'rb'))
plt.plot([i for i in range(len(ld2000_zon[0]))], np.mean(ld2000_zon, axis=0), label=r'$N=2000$', color='C6')


truth_zon, truth_mer = pickle.load(open(f'decoding_experiment010--Spectrum_of_VAE_of_truth-data/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_TRUTH_field_psds.pkl', 'rb'))
plt.plot([i for i in range(len(truth_zon[0]))], np.mean(truth_zon, axis=0), color='k', label=r'Truth')

# plt.fill_between([i for i in range(len(ld100_zon[0]))], np.min(ld100_zon, axis=0), np.max(ld100_zon, axis=0))
# plt.plot([i for i in range(len(ld100_zon[0]))], np.min(ld100_zon, axis=0), color='C0')
# plt.plot([i for i in range(len(ld100_zon[0]))], np.max(ld100_zon, axis=0), color='C0')
plt.yscale('log')
plt.legend()
plt.title(f'Zonal spectrum, {args.year}-{args.month:02d}-{args.day:02d}')
plt.ylabel(f'Mean of {args.ensemble} ensemble members')
plt.xlabel('Wavenumber')
plt.grid(linewidth=0.6, linestyle=':', color='gray')
#plt.xlim(right= max([i for i in range(len(ld200_zon[0]))]))
plt.xscale('log')

#plt.subplot(1,2,2)
plt.clf()
plt.cla()
plt.plot([i for i in range(len(ld100_mer[0]))], np.mean(ld100_mer, axis=0), label=r'$N=100$')
plt.plot([i for i in range(len(ld50_mer[0]))], np.mean(ld50_mer, axis=0), label=r'$N=50$')
plt.plot([i for i in range(len(ld200_mer[0]))], np.mean(ld200_mer, axis=0), label=r'$N=200$')
plt.plot([i for i in range(len(ld2000_mer[0]))], np.mean(ld2000_mer, axis=0), label=r'$N=2000$', color='C6')
plt.plot([i for i in range(len(truth_zon[0]))], np.mean(truth_mer, axis=0), label=r'Truth', color='k')
plt.yscale('log')
plt.legend()
plt.title(f'Temperature spectrum, {args.year}-{args.month:02d}-{args.day:02d}')
plt.ylabel(f'Mean of {args.ensemble} ensemble members')
plt.xlabel(r'Total wavenumber $n$')
plt.grid(linewidth=0.6, linestyle=':', color='gray')
#plt.xlim(right=max([i for i in range(len(ld200_mer[0]))]))
plt.xscale('log')

plt.tight_layout()

plt.savefig(f'decoding_experiment010--Spectrum_of_VAE_of_truth-figures/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_ensemble={args.ensemble}.pdf', dpi=300)
plt.savefig(f'decoding_experiment010--Spectrum_of_VAE_of_truth-figures/de010--Spectrum_of_VAE_of_truth-{args.year}-{args.month:02d}-{args.day:02d}_ensemble={args.ensemble}.jpg', dpi=300)