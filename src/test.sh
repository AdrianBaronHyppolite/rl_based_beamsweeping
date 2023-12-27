#!/bin/bash
#SBATCH --job-name=DDQN_runs
#SBATCH --output=DDQN_run_%j.out
#SBATCH --error=DDQN_run_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

# Activate virtual environment
#source your_virtualenv_name/bin/activate

# Initial value for env.ptx_dbm
ptx_dbm=15

# Run the script 3 times, each time increasing env.ptx_dbm by 10
for i in {1..3}
do
    python ARCscript.py --env.ptx_dbm=$ptx_dbm
    ptx_dbm=$((ptx_dbm + 10)) # Increase env.ptx_dbm by 10
done
