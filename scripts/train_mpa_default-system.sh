#!/bin/bash

#SBATCH --job-name=mpa_default-system        # Job name
#SBATCH -o /mnt/nas/suehyun/axolotl/logs/mpa/default-system/out_%x.txt              # Path to output log file (%j expands to job name)
#SBATCH -e /mnt/nas/suehyun/axolotl/logs/mpa/default-system/err_&x.err              # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ         # Partition name
#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Request one task (default)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --time=24:00:00            # Time limit
#SBATCH --gres=gpu:4               # Number of GPUs to be allocated

srun accelerate launch -m axolotl.cli.train examples/mpa/default-system/mistral-7b.yml