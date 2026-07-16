import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--month", help="Month", type=int, required=True)
args = parser.parse_args()

month = args.month
qty = 'T850'

with open(f'slurm_{month}.sh', 'w') as f:
    print(f'#!/bin/bash\n\
#SBATCH --nodes=1\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=09:00:00\n\
#SBATCH --mem=2G\n\
#SBATCH --cpus-per-task=2\n\
#SBATCH --partition=rude\n\
#SBATCH --qos=rude\n\
#SBATCH --error=tt{month}.err\n\
#SBATCH --output=tt{month}.out\n\
#SBATCH --job-name=tt{month}\n\
\n\
\n\
month={month}\n\
qty={qty}\n\
\n\
srun echo "This month is $month"\n\
srun echo "This quantity is $qty"\n\
\n\
srun echo "Creating bash script for climatology"\n\
srun echo "srun python BP/Proxy_20CR/data/prepare_training_tensors_ERA5_{qty}_regridded/make_climatology/make_climatology_month.py --month=$month"\n\
srun python BP/Proxy_20CR/data/prepare_training_tensors_ERA5_{qty}_regridded/make_climatology/make_climatology_month.py --month=$month\n\
\n\
srun echo "Running bash script for climatology"\n\
srun bash run_climatology_$month.sh\n\
\n\
srun echo $(date)\n\
\n\
srun echo "Creating bash script for variability climatology"\n\
srun python BP/Proxy_20CR/data/prepare_training_tensors_ERA5_{qty}_regridded/make_variability_climatology/make_variability_climatology_month.py --month=$month\n\
\n\
srun echo "Running bash script for variability climatology"\n\
srun bash run_variability_climatology_$month.sh\n\
\n\
srun echo $(date)\n\
\n\
\n\
srun echo "Creating bash script for training tensors"\n\
srun python BP/Proxy_20CR/data/prepare_training_tensors_ERA5_{qty}_regridded/make_training_tensor_month.py --month=$month\n\
\n\
srun echo "Running bash script for training tensors"\n\
srun bash run_training_tensor_$month.sh\n\
\n\
srun echo $(date)', file=f)
