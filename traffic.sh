#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --mail-user=iaroslav.kriuchkov@aalto.fi
#SBATCH --mail-type=ALL

module load anaconda
cd $WRKDIR/iarotraffic
python traffic.py
