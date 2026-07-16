#!/usr/bin/env python

# Plot the learning curve up to epoch args.epoch

import tensorflow as tf
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=103)
#args = parser.parse_args()

import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.patches as patches
from matplotlib import transforms

training_types = ['0000', '1000', '2000', '3000', '4000']
epoch_ranges = {'0000':100, '1000':100, '2000':100, '3000':100, '4000':100}
num_of_files_per_training_type = [1310, 1318, 1318, 1318, 1318, 1318, 1318, 1311, 1310, 1310]
loss_files = []
full_loss_files = []
deterministic_multipliers = []
epochss = []

def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """https://stackoverflow.com/questions/33159134/matplotlib-y-axis-label-with-multiple-colors
    @DanHickstein
    this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors[::-1]) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.15, 0.05), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)

for training_type in training_types:
    fig, ax = plt.subplots(figsize=(6, 4))

    loss_file = pickle.load(open('../models_by_epochs/Epoch_%04d/history.pkl' % (int(training_type) + epoch_ranges[training_type]), 'rb'))
    loss_files.append(loss_file)
    loss = np.array([tf.get_static_value(i) for i in loss_file['loss']])
    val_loss = np.array([tf.get_static_value(i) for i in loss_file['val_loss']])
    lr = loss_file['learning_rate']

    these_full_loss_files = [pickle.load(open('../models_by_epochs/Epoch_%04d/history_%d.pkl' % (int(training_type) + epoch_ranges[training_type], itv), 'rb')) for itv in range(0,9+1)]
    full_loss_file = dict()
    loss_values = np.array([[tf.get_static_value(i)*num_of_files_per_training_type[j] for i in these_full_loss_files[j]['full_loss']] for j in range(0,9+1)])
    full_loss_value = np.sum(loss_values, axis=0)/np.sum(num_of_files_per_training_type)
    full_loss_file['full_loss'] = full_loss_value
    loss = full_loss_value

    for key in these_full_loss_files[0].keys():
        if key != 'full_loss':
            qty_values = np.array([[i*num_of_files_per_training_type[j] for i in these_full_loss_files[j][key]] for j in range(0,9+1)])
            full_qty = np.sum(qty_values, axis=0)/np.sum(num_of_files_per_training_type)
            full_loss_file[key] = full_qty
    full_loss_files.append(full_loss_file)

    epochs = np.linspace(1,epoch_ranges[training_type],len(loss))
    cl = 'b'
    #ax.set_ylim(100, max(max(loss), max(val_loss)))
    ax.plot(epochs, loss, label='loss', color=cl, linestyle='-')
    ax.plot(epochs, val_loss, label='val_loss', color=cl, linestyle='--')
    ax.legend(loc='upper left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss value', color=cl)
    ax.set_xlim(0,epoch_ranges[training_type])
    ax.set_xticks([0] + [e for e in epochs if e%10==0])
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelcolor=cl, which='both')

    ax.grid(axis='x', linewidth=0.6, color='k', linestyle=':')
    ax.grid(axis='y', linewidth=0.6, color=cl, linestyle=':')
    ax.axvline(epochs[np.argmin(val_loss)], color='brown', linestyle='--', linewidth=0.7)
    ax.axhline(val_loss[np.argmin(val_loss)], color='brown', linestyle='--', linewidth=0.7)
    #print(epochs[np.argmin(val_loss)], val_loss[np.argmin(val_loss)], val_loss[20])

    clr = 'r'
    ax1 = ax.twinx()
    ax1.plot(epochs, lr, color=clr, label='learning rate')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Learning rate', color=clr)
    ax1.set_yscale('log')
    ax1.grid(axis='y', linewidth=0.6, color=clr, linestyle=':')
    ax1.tick_params(axis='y', labelcolor=clr)

    deterministic_multiplier = pickle.load(open('../models_by_epochs/deterministic_multipliers.pkl', 'rb'))[training_type]
    plt.title('Huber cosine multiplier: %.0e' % deterministic_multiplier)
    deterministic_multipliers.append(deterministic_multiplier)
    epochss.append(epochs)

    plt.tight_layout()
    plt.savefig(f"learning_curve_training_type={training_type}.pdf")
#
#     plt.cla()
#     fig, ax = plt.subplots(figsize=(6,4))
#     cl = 'C0'
#     cvl = 'C1'
#     rmse, val_rmse, logpz, val_logpz, logqz_x, val_logqz_x = np.array(loss_file['mse']), np.array(loss_file['val_mse']), np.array(loss_file['logpz']), np.array(loss_file['val_logpz']), np.array(loss_file['logqz_x']), np.array(loss_file['val_logqz_x'])
#
#     #ax.plot(epochs[1:], loss[1:] - loss[:-1], color='k', linestyle='-', label='change in loss')
#     ax.plot(epochs[1:], deterministic_multiplier*(rmse[1:] - rmse[:-1]) , color='k', linestyle='--', label='change in multiplied RMSE')
#     ax.plot(epochs[1:], -logpz[1:] +  logpz[:-1], color='k', linestyle='-.', label=r'change in $\log{(p(\mathbf{z}))}$')
#     ax.plot(epochs[1:], logqz_x[1:] - logqz_x[:-1] , color='k', linestyle=':', label=r'change in $\log{(q(\mathbf{z}|\mathbf{x}))}$')
#
#     #ax.plot(epochs[1:], loss[1:] - loss[:-1], color=cl, linestyle='-')
#     ax.plot(epochs[1:], deterministic_multiplier*(rmse[1:] - rmse[:-1]) , color=cl, linestyle='--')
#     ax.plot(epochs[1:], -logpz[1:] +  logpz[:-1], color=cl, linestyle='-.')
#     ax.plot(epochs[1:], logqz_x[1:] - logqz_x[:-1] , color=cl, linestyle=':')
#
#     #ax.plot(epochs[1:], val_loss[1:] - val_loss[:-1], color=cvl, linestyle='-')
#     ax.plot(epochs[1:], deterministic_multiplier*(val_rmse[1:] - val_rmse[:-1]) , color=cvl, linestyle='--')
#     ax.plot(epochs[1:], -val_logpz[1:] +  val_logpz[:-1], color=cvl, linestyle='-.')
#     ax.plot(epochs[1:], val_logqz_x[1:] - val_logqz_x[:-1] , color=cvl, linestyle=':')
#
#     multicolor_ylabel(ax, ('Training set', ',', 'Validation set'), (cl, 'k', cvl), axis='y')
#     ax.set_xticks([0] + [e for e in epochs if e%10==0])
#     ax.grid(linewidth=0.6, color='k', linestyle=':')
#     ax.set_ylim([-deterministic_multiplier/10**4,deterministic_multiplier/10**4])
#     ax.set_xlim(0,epoch_ranges[training_type])
#     ax.legend()
#
#     plt.title('Loss function attributions (RMSE multiplier: %.0e)' % deterministic_multiplier)
#     plt.savefig(f'learning_attributions_training_type={training_type}.pdf')


#AssertionError


colors = ['C0', 'C1', 'crimson', 'C6', 'linen']
colors = ['b', 'cyan', 'pink', 'gold', 'crimson']
ylabels = []
ylabelscolors = []
labelslocations = []
nsubplots = 4
fig, axs = plt.subplots(nrows=nsubplots, ncols=1, figsize=(6,13))
for i in range(len(deterministic_multipliers)):
    jsubplot = -1
    loss_file = loss_files[i]
    full_loss_file = full_loss_files[i]

    jsubplot += 1
    if i == 0:
        axs[jsubplot].plot(epochss[i], full_loss_file['full_huber_cosine'], color='k', label='training Huber cosine')
        axs[jsubplot].plot(epochss[i], loss_file['val_huber_cosine'], color='k', linestyle='--', label='validation Huber cosine')
        ylabels.append('Huber c. multipliers:')
        ylabelscolors.append('k')
        labelslocations.append((0.05, 0.95))
    axs[jsubplot].plot(epochss[i], full_loss_file['full_huber_cosine'], color=colors[i])
    axs[jsubplot].plot(epochss[i], loss_file['val_huber_cosine'], color=colors[i], linestyle='--')
    ylabels.append('%.0e' % deterministic_multipliers[i])
    ylabelscolors.append(colors[i])
    axs[jsubplot].set_yscale('log')
    axs[jsubplot].set_title('Huber cosine')

    
    jsubplot += 1
    if i == 0:
        axs[jsubplot].plot(epochss[i], full_loss_file['full_logpz'], color='k', label=r'training $\log{(p(\mathbf{z}))}$')
        axs[jsubplot].plot(epochss[i], loss_file['val_logpz'], color='k', linestyle='--', label=r'validation $\log{(p(\mathbf{z}))}$')
        labelslocations.append((0.7, 0.3))
    axs[jsubplot].plot(epochss[i], full_loss_file['full_logpz'], color=colors[i])
    axs[jsubplot].plot(epochss[i], loss_file['val_logpz'], color=colors[i], linestyle='--')
    axs[jsubplot].set_title(r'$\log{(p(\mathbf{z}))}$')
    axs[jsubplot].set_ylim(-5,0)
    axs[jsubplot].axhline(-1.4125933495324293, color='k', linestyle='-.')

    jsubplot += 1
    if i == 0:
        axs[jsubplot].plot(epochss[i], full_loss_file['full_logqz_x'], color='k', label=r'training $\log{(q(\mathbf{z}|\mathbf{x}))}$')
        axs[jsubplot].plot(epochss[i], loss_file['val_logqz_x'], color='k', linestyle='--', label=r'validation $\log{(q(\mathbf{z}|\mathbf{x}))}$')
        labelslocations.append((0.05, 0.95))
    axs[jsubplot].plot(epochss[i], full_loss_file['full_logqz_x'], color=colors[i])
    axs[jsubplot].plot(epochss[i], loss_file['val_logqz_x'], color=colors[i], linestyle='--')
    axs[jsubplot].set_title(r'$\log{(q(\mathbf{z}|\mathbf{x}))}$')
    axs[jsubplot].axhline(-1.4125933495324293, color='k', linestyle='-.')

    jsubplot += 1
    if i == 0:
        axs[jsubplot].plot(epochss[i], full_loss_file['full_mse'], color='k', label='training MSE')
        axs[jsubplot].plot(epochss[i], loss_file['val_mse'], color='k', linestyle='--', label='validation MSE')
        labelslocations.append((0.05, 0.95))
    axs[jsubplot].plot(epochss[i], full_loss_file['full_mse'], color=colors[i])
    axs[jsubplot].plot(epochss[i], loss_file['val_mse'], color=colors[i], linestyle='--')
    axs[jsubplot].set_yscale('log')
    axs[jsubplot].set_title('MSE')

max_epoch = np.amax([np.amax(es) for es in epochss])

for jsubplot in range(nsubplots):
    trans_x = transforms.blended_transform_factory(axs[jsubplot].transAxes, axs[jsubplot].transData)
    trans_y = transforms.blended_transform_factory(axs[jsubplot].transData, axs[jsubplot].transAxes)

    axs[jsubplot].set_xlim(0, max_epoch)
    axs[jsubplot].set_xlabel('Epoch')
    axs[jsubplot].grid(linestyle=':', linewidth=0.7, color='grey')
    axs[jsubplot].legend()
    axs[jsubplot].set_xticks([e for e in range(int(max_epoch) + 1) if e%10==0])
    #multicolor_ylabel(axs[jsubplot], ylabels, ylabelscolors, axis='y')
    top_location = labelslocations[jsubplot]
    max_locy = top_location[1] + 0.01
    min_locx = top_location[0] - 0.01
    for jlabel in range(len(ylabels)):
        locx = top_location[0]
        locy = top_location[1] - 0.05*(jlabel)
        min_locy = locy - 0.05
        axs[jsubplot].text(
            x=locx,
            y=locy,
            s=ylabels[jlabel],
            horizontalalignment="left",
            verticalalignment="top",
            color=ylabelscolors[jlabel],
            transform = axs[jsubplot].transAxes,
            zorder=1000
        )

    rect = plt.Rectangle((min_locx, min_locy), 0.26, max_locy - min_locy, transform=transforms.blended_transform_factory(trans_x, trans_y),  facecolor='w', alpha=0.8, zorder=999)

    axs[jsubplot].add_patch(rect)



print(ylabels)
print(ylabelscolors)

plt.tight_layout()
plt.savefig('learning_metrics.pdf')