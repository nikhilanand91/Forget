#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --time=00:20:00                        # The job will run for 10 hours
#SBATCH -o /home/mila/n/nikhil.anand/ForgetTest/slurm-%j.out  # Write the log on tmp1

module --quiet load python/3.7
module load pytorch
virtualenv $SLURM_TMPDIR/
source $SLURM_TMPDIR/bin/activate
pip install torch torchvision
pip install Pillow

python /home/mila/n/nikhil.anand/ForgetExperiments/Forget/main/run.py

echo "Done executing!" > "test.log"

cp -r $SLURM_TMPDIR/ /home/mila/n/nikhil.anand/ForgetExperiments/