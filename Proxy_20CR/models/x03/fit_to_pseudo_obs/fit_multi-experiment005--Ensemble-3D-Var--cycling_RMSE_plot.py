#!/usr/bin/env python
import time

import numpy as np

import datetime

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import argparse
import pickle


start = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=1020)
parser.add_argument(
    "--ensemble", help="No. of ensemble members", type=int, required=False, default=150
)
parser.add_argument('--plot', help="Plot", default=False, action=argparse.BooleanOptionalAction) #In order not to plot: --no-plot
parser.add_argument('--std_first_multiplier', help="Multiplier of std of first guess", type=float, required=False, default=1.0)
parser.add_argument('--obs_std', help="Standard deviation of pseudo observations, degree C", type=float, required=False, default=0.0)
parser.add_argument('--minimization_learning_rate', help='Learning rate for ADAM optimizer when performing minimization in latent space', type=float, required=False, default=0.01)
parser.add_argument('--adaptive_lr', help='Whether the learning rate for ADAM optimizer when performing minimization in latent space decreases if loss is on plateau or not', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--perfect_obs', help='Perfect observations', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--perfect_first', help='Perfect first guess', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--custom_addon', type=str, default='', required=False)
parser.add_argument('--obs_increment', help="Observation increment for single observation experiment (if not 0.0, else the value is sampled from the 'truth' field)", type=float, required=False, default=0.0)
parser.add_argument('--singobs_lat', help="Latitude in case of single observation experiment", type=float, required=False, default=False)
parser.add_argument('--singobs_lon', help="Latitude in case of single observation experiment", type=float, required=False, default=False)
parser.add_argument('--diagonal_B', help='Only use diagonal elements of B-matrix (no correlations between latent elements)', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--flow_B_fraction', help="Fraction of B-matrix, computed from ensemble members of ERA5", type=float, required=False, default=0.0)
parser.add_argument('--ERA5_ens_mem_spread_multiplicator', help="The multiplicator of the ensemble spread from ERA5 ensemble members", type=int, required=False, default=1)
parser.add_argument('--plot_singles', help="Plot some figures one by one (only if --plot)", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--cycling_starting_point', help='Date for the first cycle', type=str, default='', required=False)

args = parser.parse_args()

res = 4.0
res_str = str(res)

rmse_filename = f'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-epoch={args.epoch}_obs_std={args.obs_std}_res={res_str}_ensemble={args.ensemble}'
if args.std_first_multiplier != 1.0:
    rmse_filename += f'std_first_mutiplier={args.std_first_multiplier}'
if args.minimization_learning_rate != 0.01:
    rmse_filename += f'minimization_lr={args.minimization_learning_rate}'
if not args.adaptive_lr:
    rmse_filename += '_no_adaptive_lr'
if args.perfect_first:
    rmse_filename += '_perfect_first'
if args.perfect_obs:
    rmse_filename += '_perfect_obs'
if args.diagonal_B:
    rmse_filename += '_diagonal_B'
if args.flow_B_fraction > 0:
    rmse_filename += f'_flow_B_fraction={args.flow_B_fraction}'
    if args.ERA5_ens_mem_spread_multiplicator != 1:
        rmse_filename += f'_ERA5_ens_spr_mul={args.ERA5_ens_mem_spread_multiplicator}'
if args.cycling_starting_point != '':
    rmse_filename += f'_cycling_from_{args.cycling_starting_point}'
if len(args.custom_addon) > 0:
    rmse_filename += '_' + args.custom_addon

override = True
if override:
    rmse_filename = 'fit_multi-experiment005--Ensemble-3D-Var-data/fe005--Ensemble-3D-Var-epoch=1020_obs_std=1.0_res=4.0_ensemble=150minimization_lr=0.1_diagonal_B_cycling_from_2019-4-15'
    print('\nOVERRODE RMSE_FILENAME!!!\n')
    time.sleep(5)


rmse_dict = pickle.load(open(rmse_filename + '.pkl', 'rb'))

print(rmse_dict.keys())

all_dates = list(rmse_dict.keys())
all_dates.sort()

all_dates_datetime = [datetime.date(year=int(i[:4]), month=int(i[5:7]), day=int(i[8:10])) for i in all_dates]
cycle_no = [(i - min(all_dates_datetime)).days + 1 for i in all_dates_datetime]

print(cycle_no)
print([str(i) for i in all_dates])
print(rmse_dict['2019-04-18'])

plt.subplot(2, 1, 1)
plt.plot(cycle_no, [rmse_dict[str(i)]['rmse_background_gp'] for i in all_dates], label='Background')
plt.plot(cycle_no, [rmse_dict[str(i)]['rmse_analysis_gp'] for i in all_dates], label='Analysis')
plt.legend(loc='upper right')
plt.xlabel('Cycle')
plt.ylabel('RMSE [K]')
plt.title('RMSE in the grid point space')
plt.xlim(0, max(cycle_no))
plt.grid(linestyle=':')

plt.subplot(2, 1, 2)
# Forgot to apply sqrt in latent space error, correcting this here
plt.plot(cycle_no, [np.sqrt(rmse_dict[str(i)]['rmse_background_latent']) for i in all_dates], label='Background', linestyle='--')
plt.plot(cycle_no, [np.sqrt(rmse_dict[str(i)]['rmse_analysis_latent']) for i in all_dates], label='Analysis', linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('Cycle')
plt.ylabel('RMSE')
plt.title('RMSE in the latent space')
plt.grid(linestyle=':')
plt.ylim(bottom=0)
plt.xlim(0, max(cycle_no))

plt.tight_layout()

if override:
    plt.savefig('fit_multi-experiment005--Ensemble-3D-Var-figures/cycling_override_final.pdf', dpi=300)
else:
    plt.savefig('fit_multi-experiment005--Ensemble-3D-Var-figures/' + rmse_filename[len('fit_multi-experiment005--Ensemble-3D-Var-data/'):] + '.jpg', dpi=300)


fig, ax1 = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(6)

c1 = 'C0'
c2 = 'C2'

ax1.plot(cycle_no, [rmse_dict[str(i)]['rmse_background_gp'] for i in all_dates], color=c1, linestyle='-')#, label='Background')
ax1.plot(cycle_no, [rmse_dict[str(i)]['rmse_analysis_gp'] for i in all_dates], color=c1, linestyle='--')#, label='Analysis')
#ax1.legend(loc='upper right')
ax1.set_xlabel('Cycle')
ax1.set_ylabel('Grid-point-space RMSE [K]', color=c1)
ax1.set_xlim(0, max(cycle_no))
ax1.set_ylim(0,3)
ax1.grid(linestyle=':', axis='x')
ax1.tick_params(axis='y', labelcolor=c1)
ax1.plot([0,0], [0,0], 'k-', label='Background')
ax1.plot([0,0], [0,0], 'k--', label='Analysis')
ax1.legend(loc='lower right')


ax2 = ax1.twinx()
ax2.plot(cycle_no, [np.sqrt(rmse_dict[str(i)]['rmse_background_latent']) for i in all_dates], color=c2, linestyle='-')
ax2.plot(cycle_no, [np.sqrt(rmse_dict[str(i)]['rmse_analysis_latent']) for i in all_dates], color=c2, linestyle='--')
#ax1.legend(loc='upper right')
ax2.set_ylabel('Latent-space RMSE', color=c2)
ax2.set_ylim(0,1)
#ax1.grid(linestyle=':')
ax2.tick_params(axis='y', labelcolor=c2)

print([np.sqrt(rmse_dict[str(i)]['rmse_background_latent']) for i in all_dates][0])
print([np.sqrt(rmse_dict[str(i)]['rmse_analysis_latent']) for i in all_dates][0])


fig.tight_layout()

if override:
    plt.savefig('fit_multi-experiment005--Ensemble-3D-Var-figures/cycling_override_final2.pdf', dpi=300)