#!/usr/bin/env python

# Read in a field of ERA5 T850 as an Iris cube.
# Convert it into an anomaly
# It is 1440x720 pixels
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

import tensorflow as tf
import numpy as np

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
import dask

dask.config.set(scheduler="single-threaded")


import IRData.twcr as twcr
import iris
import datetime
import argparse
import os
import sys

sys.path.append("%s" % os.path.dirname(__file__))
from ERA5_load import ERA5_load_T850
from ERA5_load import ERA5_load_T850_climatology
from ERA5_load import ERA5_load_T850_variability_climatology
from ERA5_load import ERA5_roll_longitude
from ERA5_load import ERA5_trim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument("--training_section", help="Integer for the training section 1-9", type=int, default=0, required=False)
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
parser.add_argument('--experiment', help='Experiment number', type=int, default=None, required=False)
parser.add_argument("--validation", help="validation data, not training", action="store_true")
args = parser.parse_args()

if args.experiment is None: # Original (well, slightly modified) version
    if args.opfile is None:
        if args.test:
            purpose = "test"
        elif args.training_section == 0:
            purpose = "training"
        else:
            purpose = "training%d" % args.training_section
        args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/%s/%s/%04d-%02d-%02d.tfd") % (
            os.getenv("SCRATCH"),
            "ERA5",
            "daily_T2m",
            purpose,
            args.year,
            args.month,
            args.day,
        )

elif args.experiment == 1:
    if args.opfile is None:
        if args.test:
            purpose = "x01test"
        elif args.validation:
            purpose = "x01validation"
        else:
            purpose = "x01training%d" % args.training_section
        args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/%s/%s/%04d-%02d-%02d.tfd") % (
            os.getenv("SCRATCH"),
            "ERA5",
            "daily_T2m",
            purpose,
            args.year,
            args.month,
            args.day,
        )

elif args.experiment == 2:
    if args.opfile is None:
        if args.year > 2018:
            purpose = "x02test"
        elif args.year > 2014:
            purpose = "x02validation"
        else:
            purpose = "x02training%d" % args.training_section
        args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/%s/regridded_version/%s/%04d-%02d-%02d.tfd") % (
            os.getenv("SCRATCH"),
            "ERA5",
            "daily_T2m",
            purpose,
            args.year,
            args.month,
            args.day,
        )

elif args.experiment == 3:
    if args.opfile is None:
        if args.year > 2018:
            purpose = "x03test"
        elif args.year > 2014:
            purpose = "x03validation"
        else:
            purpose = "x03training%d" % args.training_section
        args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/%s/regridded_version/%s/%04d-%02d-%02d.tfd") % (
            os.getenv("SCRATCH"),
            "ERA5",
            "daily_T850",
            purpose,
            args.year,
            args.month,
            args.day,
        )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

if args.experiment is None or args.experiment == 1:
    # Load and anomalise data
    t = ERA5_load_T2m(args.year, args.month, args.day)
    c = ERA5_load_T2m_climatology(args.year, args.month, args.day)
    t = t - c
    # Rescale to range 0-1 (approx)
    t /= 15
    t += 0.5
    # Roll the longitude to put the UK in the centre
    t = ERA5_roll_longitude(t)
    # discard bottom left to make sizes multiply divisible by 2
    t = ERA5_trim(t)

elif args.experiment == 2:
    #Load and standardise data
    t = ERA5_load_T2m(args.year, args.month, args.day)
    c = ERA5_load_T2m_climatology(args.year, args.month, args.day)
    vc = ERA5_load_T2m_variability_climatology(args.year, args.month, args.day)
    t = t - c
    t = t / vc
    # Roll the longitude to put the UK in the centre
    t = ERA5_roll_longitude(t)

elif args.experiment == 3:
    #Load and standardise data
    t = ERA5_load_T850(args.year, args.month, args.day)
    c = ERA5_load_T850_climatology(args.year, args.month, args.day)
    vc = ERA5_load_T850_variability_climatology(args.year, args.month, args.day)
    t = t - c
    t = t / vc
    # Roll the longitude to put the UK in the centre
    t = ERA5_roll_longitude(t)


# Convert to Tensor
ict = tf.convert_to_tensor(t.data, np.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
