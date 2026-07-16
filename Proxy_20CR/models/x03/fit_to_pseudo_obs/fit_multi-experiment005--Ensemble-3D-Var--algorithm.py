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


import pickle

import multiprocessing


def parallel_worker(input):
    latent_background = input['latent_background']
    ob_locations = input['ob_locations']
    pseudo_obs = input['pseudo_obs']
    B_matrix_inv = input['B_matrix_inv']
    R_matrix_inv = input['R_matrix_inv']
    renormalization = input['renormalization']
    epoch = input['epoch']
    init_lr = input['init_lr']
    ensemble_member_idx = input['ensemble_member_idx']
    #print('inside parallel_worker')

    output = findLatent3DVar(latent_background=latent_background,
                             ob_locations=ob_locations,
                             pseudo_obs=pseudo_obs,
                             B_matrix_inv=B_matrix_inv,
                             R_matrix_inv=R_matrix_inv,
                             renormalization=renormalization,
                             init_lr=init_lr,
                             epoch=epoch,
                             ensemble_member_idx=ensemble_member_idx
                             )
    return output


# Find a latent state which generates a field fitted to the pseudo obs.
def findLatent3DVar(
        latent_background,  # vector 1 x autoencoder.latent_dim (already perturbed)
        ob_locations,
        pseudo_obs,  # vector 1 x number of obs. (already perturbed)
        B_matrix_inv,
        R_matrix_inv,
        renormalization,  # variability_climatology
        init_lr,  # Define your initial learning rate
        epoch,
        ensemble_member_idx, # Just to print
        max_num_steps=1000,
        factor_lr=0.5,  # Factor by which the learning rate is reduced
        patience_lr=3,   # Number of epochs with no improvement after which learning rate is reduced
        rtol_stop=0.01,   # Relative tolerance for convergence criterion
        patience_stop=10,
        minimum_lr = 1e-4
):
    #print('inside 3D-Var')
    sys.path.append("%s/.." % os.path.dirname(__file__))
    from autoencoderModel import DCVAE
    import gc

    autoencoder = DCVAE()
    #print('defined VAE')
    weights_dir = ("../models_by_epochs/" + "Epoch_%04d") % (
        epoch,
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

    #print('Converting ob_locations')
    t_lats_npy, t_lons_npy = ob_locations
    t_lats = tf.convert_to_tensor(1.0 - t_lats_npy, tf.float32)  # ker so do zdaj t_lats=180 na NP in t_lats=0 na SP,
    # medtem ko so idx v matrikah 0 na NP in 719 na SP
    t_lons = tf.convert_to_tensor(t_lons_npy, tf.float32)
    # input('end3')
    t_obs = tf.stack((t_lats * 720, t_lons * 1440), axis=1)
    t_obs = tf.expand_dims(t_obs, 0)
    ob_locations =t_obs
    #print('Got t_obs')

    #print('Converting background and observations values to tf.tensor (also B_matrix_inv and R_matrix_inv)')
    latent_background = tf.convert_to_tensor(latent_background)
    #print('pseudo obs before tensor', pseudo_obs)
    pseudo_obs = tf.convert_to_tensor(pseudo_obs)
    #print('pseudo obs as tensor', pseudo_obs)
    B_matrix_inv = tf.convert_to_tensor(B_matrix_inv, dtype=tf.float32)
    R_matrix_inv = tf.convert_to_tensor(R_matrix_inv, dtype=tf.float32)
    #print('Converted to tf.tensor')

    latent = tf.Variable(tf.identity(latent_background))
    # print('latent1')
    # from pprint import pprint
    # pprint(dir(autoencoder))
    # print(autoencoder)
    optimizer = tf.optimizers.Adam(learning_rate=init_lr)
    best_loss = float('inf')  # Variable to keep track of the best loss
    best_loss_step = 0
    best_latent = tf.identity(latent)
    num_epochs_no_improvement_lr = 0  # Variable to keep track of the number of epochs with no improvement (for learning rate reduction)
    num_epochs_no_improvement_stop = 0  # Variable to keep track of the number of epochs with too little improvement (for stopping criterion)
    losses = []
    logpzs = []
    all_Jo = []
    all_gradients = []
    ending_step = max_num_steps  # if the convergence is not reached before that
    #print('defined ending step')

    def decodeFit3DVar():
        #print('trying to decode')
        decoded = autoencoder.decode(latent)
        #print('SUCCESSFULLY USED AUTOENCODER')
        renormalized_decoded = decoded * tf.convert_to_tensor(np.reshape(renormalization, (1, 720, 1440, 1)))
        #print('shape of renormalized_decoded')
        decoded_obs = tf.squeeze(interpolate_bilinear(renormalized_decoded, ob_locations, indexing="ij"), [0, 2])
        #print('decoded_obs', decoded_obs)
        # print('do', np.shape(decoded_obs), 'ib', np.shape(interpolate_bilinear(renormalized_decoded, ob_locations, indexing="ij")))
        decoded_obs_vec = tf.reshape(decoded_obs, (decoded_obs.shape[0], 1))  # H(D(l))
        #print('decoded_obs_vec', decoded_obs_vec)
        #print('pseudo_obs', pseudo_obs)
        z_minus_zb_transposed = tf.cast(tf.subtract(latent, latent_background), dtype=tf.float32)
        z_minus_zb_vec = tf.transpose(tf.cast(z_minus_zb_transposed, dtype=tf.float32))
        H_of_D_of_z_minus_y = tf.cast(tf.subtract(pseudo_obs, decoded_obs_vec), dtype=tf.float32)
        #print('H_of_D_of_l_minus_y', H_of_D_of_z_minus_y)
        # np.matmul(B_matrix_inv, l_minus_lb)
        J_b = 1 / 2 * tf.matmul(z_minus_zb_transposed, tf.matmul(B_matrix_inv, z_minus_zb_vec))
        J_o = 1 / 2 * tf.matmul(np.transpose(H_of_D_of_z_minus_y), tf.matmul(R_matrix_inv, H_of_D_of_z_minus_y))
        J = tf.add(J_b, J_o)
        # print(tf.squeeze(J), J, J_b, J_o)

        return tf.squeeze(J), tf.squeeze(J_o)

    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
        )

    for step in range(1, max_num_steps + 1):
        #print('trying logpzs')
        #print('ae latent dim', autoencoder.latent_dim)
        logpzs.append(log_normal_pdf(latent, 0., 0.) / autoencoder.latent_dim)
        #print('got logpzs')
        with tf.GradientTape() as tape:
            loss, Jo = decodeFit3DVar()

        # print(type(loss), type(latent))
        # print(latent.shape)
        losses.append(loss)
        all_Jo.append(Jo)
        previous_latent = tf.identity(latent)
        gradients = tape.gradient(loss, [latent])
        #print(np.shape(gradients))
        all_gradients.append(tf.norm(gradients, ord='euclidean'))
        #print(losses[-1], all_gradients[-1], optimizer.learning_rate.numpy())

        optimizer.apply_gradients(zip(gradients, [latent]))

        # Check for improvement in loss (for learning rate)
        if loss < best_loss:
            best_loss = loss
            best_loss_step = step - 1
            best_latent = previous_latent
            num_epochs_no_improvement_lr = 0
        else:
            num_epochs_no_improvement_lr += 1
            # print('no improvement in step', step)

        # Check for improvement in loss (for stopping criterion)
        if step >= 2:
            if losses[-1] / losses[-2] < 1 - rtol_stop:
                num_epochs_no_improvement_stop = 0
            else:
                num_epochs_no_improvement_stop += 1


        # Check if learning rate needs to be reduced
        if num_epochs_no_improvement_lr >= patience_lr:
            current_lr = optimizer.learning_rate.numpy()
            new_lr = max(current_lr * factor_lr, minimum_lr)
            optimizer.learning_rate.assign(new_lr)
            num_epochs_no_improvement_lr = 0
            # print(step, new_lr)

        # Check for convergence
        if num_epochs_no_improvement_stop >= patience_stop:
            ending_step = step
            break

        # # Check for convergence
        # if step >= min_num_steps:
        #     if (step - best_loss_step >= step_tol or
        #             min(losses[-min_num_steps // 2:]) / min(
        #                 losses[-2 * min_num_steps // 2:-min_num_steps // 2]) > 1 - rtol_stop):
        #         # print(step, min(losses[-min_num_steps//2:]), min(losses[-2*min_num_steps//2:-min_num_steps//2]), -2*min_num_steps//2)
        #         ending_step = step
        #         break
        if step in [2, 10]:
            print('Ens. member', ensemble_member_idx, 'step', step, 'loss', loss, 'ratio', losses[-1] / losses[-2], 'nnis', num_epochs_no_improvement_stop)

    print('Ens. member', ensemble_member_idx, 'ending step', ending_step, 'ending loss', loss, 'best step', best_loss_step, 'best loss', best_loss)

    del autoencoder
    del t_lats
    del t_lons
    del t_obs
    del R_matrix_inv
    del B_matrix_inv
    del latent_background
    del pseudo_obs
    gc.collect()

    return {'best_latent': best_latent, 'best_loss': best_loss, 'losses': losses, 'logpzs': logpzs, 'all_Jo':all_Jo, 'all_gradients':all_gradients}


if __name__ == '__main__':
    inputs = pickle.load(open('fit_multi-experiment005--Ensemble-3D-Var-data/algorithm_inputs.pkl', 'rb'))
    inputs_for_ensemble_3D_Var = inputs['inputs_for_ensemble_3D_Var']
    cpus = inputs['cpus']
    file_to_dump = inputs['name']
    obs_gp_std = inputs['obs_gp_std']

    # inputs_for_ensemble_3D_Var = [{'latent_background': 0,
    #                                'ob_locations': 0,
    #                                'pseudo_obs': 0,
    #                                'renormalization': 0,
    #                                'B_matrix_inv': 0,
    #                                'R_matrix_inv': 0,
    #                                'init_lr': 0,
    #                                'epoch':1020,
    #                                'num_steps': 1000,
    #                                'factor_lr': 0.5,
    #                                'patience_lr': 3,
    #                                'step_tol': 20,
    #                                'rtol_stop': 0.001,
    #                                'min_num_steps': 100} for iens_mem in range(50)]

    ensemble = len(inputs_for_ensemble_3D_Var)
    print('Ensemble size:', ensemble)
    print('Requested cpus:', cpus)
    print('Dedicated cpus:', min(cpus, ensemble))
    time.sleep(1)

    pool = multiprocessing.Pool(processes=min(cpus, ensemble))
    print('defined pool')
    results = pool.map(parallel_worker, inputs_for_ensemble_3D_Var)
    print('results!')
    pool.close()
    pool.join()

    latent1 = [results[iens_mem]['best_latent'] for iens_mem in range(len(results))]
    print('shape latent1', np.shape(latent1))
    print('shape inputs_for...', np.shape(inputs_for_ensemble_3D_Var[0]['latent_background']), '[-1]', np.shape(inputs_for_ensemble_3D_Var[0]['latent_background'])[-1])
    latent = tf.convert_to_tensor(np.reshape([results[iens_mem]['best_latent'] for iens_mem in range(len(results))], (ensemble, np.shape(inputs_for_ensemble_3D_Var[0]['latent_background'])[-1])), dtype=tf.float32)
    print('shape latent', np.shape(latent.numpy()))
    best_loss = [results[iens_mem]['best_loss'] for iens_mem in range(len(results))]
    losses = [results[iens_mem]['losses'] for iens_mem in range(len(results))]
    logpzs = [results[iens_mem]['logpzs'] for iens_mem in range(len(results))]
    all_Jo = [results[iens_mem]['all_Jo'] for iens_mem in range(len(results))]
    all_gradients = [results[iens_mem]['all_gradients'] for iens_mem in range(len(results))]

    previous_latents = tf.convert_to_tensor(np.reshape([inputs_for_ensemble_3D_Var[iens_mem]['latent_background'] for iens_mem in range(len(results))], (ensemble, np.shape(inputs_for_ensemble_3D_Var[0]['latent_background'])[-1])), dtype=tf.float32)
    print('np.shape(inputs_for_ensemble_3D_Var[0][pseudo_obs])',np.shape(inputs_for_ensemble_3D_Var[0]['pseudo_obs']))
    pseudo_obs = tf.convert_to_tensor([inputs_for_ensemble_3D_Var[iens_mem]['pseudo_obs'] for iens_mem in range(len(results))], dtype=tf.float32)
    print('shape previous latents', np.shape(previous_latents))
    print('shape pseudo obs', np.shape(pseudo_obs))

    t_lats_npy, t_lons_npy = inputs_for_ensemble_3D_Var[0]['ob_locations']
    t_lats = tf.convert_to_tensor(1.0 - t_lats_npy, tf.float32)  # ker so do zdaj t_lats=180 na NP in t_lats=0 na SP,
    # medtem ko so idx v matrikah 0 na NP in 719 na SP
    t_lons = tf.convert_to_tensor(t_lons_npy, tf.float32)
    # input('end3')
    t_obs = tf.stack((t_lats * 720, t_lons * 1440), axis=1)
    t_obs = tf.expand_dims(t_obs, 0)


    comment = "latent: latent element values after fit (initial guess was sampled from true distribution of latent values of previous day, i.e. previous_latent)\n" + \
              "t_obs: observations' locations\n" + \
              "previous_latent: samples, used as initial guess for fit\n" + \
              "pseudo_obs: pseudo observations in fit\n" + \
              "obs_gp_std: standard deviation of observations\n" + \
              "best_loss: best loss for each ensemble member\n" + \
              "losses: all loss values for each ensemble member\n" + \
              "logpzs: all logpz values for each ensemble member\n" + \
              "all_Jo: all loss values for each ensemble member (observation term only)\n" + \
              "all_gradient: all gradient of loss function values for each ensemble member (Euclid norm)\n"

    dict_to_dump = {'latent': latent, 't_obs': t_obs,
                    'previous_latent': previous_latents, 'obs_gp_std': obs_gp_std,
                    'best_loss': best_loss, 'loss': losses, 'logpzs': logpzs,
                    'all_Jo':all_Jo, 'all_gradients':all_gradients,
                    'pseudo_obs':pseudo_obs,
                    'comment': comment}

    pickle.dump(dict_to_dump, open(file_to_dump + '.pkl', 'wb'))