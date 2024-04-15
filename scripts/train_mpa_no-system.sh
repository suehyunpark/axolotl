#!/bin/bash

#SBATCH --job-name=mpa_mistral-7b-no-system        # Job name
#SBATCH -o /mnt/nas/suehyun/axolotl/logs/out_%x.txt              # Path to output log file (%j expands to job name)
#SBATCH -e /mnt/nas/suehyun/axolotl/logs/err_%x.err              # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ         # Partition name
#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Request one task (default)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --time=24:00:00            # Time limit
#SBATCH --gres=gpu:4               # Number of GPUs to be allocated

export WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
export HF_TOKEN="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"

# echo "Starting job"
srun accelerate launch --main_process_port 0 -m axolotl.cli.train /mnt/nas/suehyun/axolotl/examples/mpa/no-system/mistral-7b.yml