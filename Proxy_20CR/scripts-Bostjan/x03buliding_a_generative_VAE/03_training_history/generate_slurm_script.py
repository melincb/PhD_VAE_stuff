import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--start", help="Starting epoch", type=int, required=True)
parser.add_argument("--end", help="Ending epoch", type=int, required=True)
parser.add_argument("--training_section", help="Training section (0-9)", type=int, required=True)
args = parser.parse_args()



with open(f'slurm_{args.start:04d}_{args.end:04d}_ts{args.training_section}.sh', 'w') as f:
    print(f'#!/bin/bash\n\
#SBATCH --nodes=1\n\
#SBATCH --ntasks=1\n\
#SBATCH --time=24:00:00\n\
#SBATCH --mem=10G\n\
#SBATCH --cpus-per-task=1\n\
#SBATCH --partition=rude\n\
#SBATCH --qos=rude\n\
#SBATCH --error=the_full_history_{args.start}_{args.end}_ts{args.training_section}.err\n\
#SBATCH --output=the_full_history_{args.start}_{args.end}_ts{args.training_section}.out\n\
#SBATCH --job-name=full_history_{args.start}_{args.end}_ts{args.training_section}\n\
\n\
\n\
\n\
srun echo "Running python script (from its directory)!"\n\
cd BP/Proxy_20CR/models/x03/models_by_epochs\n\
srun echo "srun python BP/Proxy_20CR/models/x03/models_by_epochs/full_history.py --start={args.start} --end={args.end} --training_section={args.training_section}"\n\
srun python full_history.py --start={args.start} --end={args.end} --training_section={args.training_section}\n\
\n\
srun echo $(date)', file=f)
