#!/bin/bash
#SBATCH --job-name=fidelity         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=6        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=yuhan.liu@mpq.mpg.de

module purge
module load anaconda/3/2023.03
source activate quimb_env

# $P1 = N, $P2 = floor, $P3 = Delta (optional; the script defaults to 0.5)
python multilayer_optimizarion.py $P1 $P2 $P3
