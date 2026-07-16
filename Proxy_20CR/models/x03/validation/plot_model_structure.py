#!/usr/bin/env python

# Plot the learning curve up to epoch args.epoch

import tensorflow as tf
import numpy as np
import pickle
import argparse
import os
import sys

parser = argparse.ArgumentParser()
#parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=103)
#args = parser.parse_args()

import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.patches as patches
from matplotlib import transforms

sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
            3050,
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

tf.keras.utils.plot_model(autoencoder)