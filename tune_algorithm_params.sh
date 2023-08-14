#!/bin/bash

#SBATCH -J tune_algorithm_params
## Specify the name of the run.
#SBATCH -a 0-2
## Controls the number of replications of the job that are run
## The specific ID of the replication can be accesses with the environment variable $SLURM_ARRAY_TASK_ID
## Can be used for seeding
#SBATCH -n 1  
## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1
## Specify the number of cores. Leave this value at 1.
#SBATCH --mem-per-cpu 2000
## Here you can control the amount of memory that will be allocated for you job. To set this,
## you should run the programm on your local computer first and observe the memory it consumes.
#SBATCH -t 03:00:00
## Do not allocate more time than you REALLY need. Maximum is 6 hours.

#SBATCH -A kurs00067                                                                                                                                                                                                    
#SBATCH -p kurs00067                                                                                                                                                                                                    
#SBATCH --reservation=kurs00067

#SBATCH -o "C:/Users/Timo/Desktop/HW3_QUESTIONS/logs/sbatch/%A_%a.out
#SBATCH -e "C:/Users/Timo/Desktop/HW3_QUESTIONS/logs/sbatch/%A_%a.err
## <your path> refers to the directory where you would like your outputs. Be sure to create folders "logs/results" and "logs/sbatch" in 
## <your path> before running this script

results_dir=<your path>/logs/results
python tune_algorithm_params.py --results_dir $results_dir --seed $SLURM_ARRAY_TASK_ID --algorithm <algorithm> --hyperparam_value <value>
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."