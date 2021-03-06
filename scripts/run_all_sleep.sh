#!/bin/bash
NCORES=16
NWORKERS=16
SORTED_WALLTIME='5:00:00'
CLUSTERLESS_WALLTIME='14:00:00'
REMY_WALLTIME='24:00:00'

# Clusterless
python queue_sleep_cluster_jobs.py --data_type 'clusterless' \
                                   --n_cores $NCORES \
                                   --n_workers $NWORKERS \
                                   --wall_time $CLUSTERLESS_WALLTIME \
                                   --overwrite \

python queue_sleep_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 3 \
                                   --data_type 'clusterless' \
                                   --n_cores $NCORES \
                                   --n_workers $NWORKERS \
                                   --wall_time $REMY_WALLTIME \
                                   --overwrite \

python queue_sleep_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 5 \
                                   --data_type 'clusterless' \
                                   --n_cores $NCORES \
                                   --n_workers $NWORKERS \
                                   --wall_time $REMY_WALLTIME \
                                   --overwrite \

python queue_sleep_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 3 \
                                   --data_type 'clusterless' \
                                   --n_cores $NCORES \
                                   --n_workers $NWORKERS \
                                   --wall_time $REMY_WALLTIME \
                                   --overwrite \

python queue_sleep_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 5 \
                                   --data_type 'clusterless' \
                                   --n_cores $NCORES \
                                   --n_workers $NWORKERS \
                                   --wall_time $REMY_WALLTIME \
                                   --overwrite \

python queue_sleep_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 3 \
                                   --data_type 'clusterless' \
                                   --n_cores $NCORES \
                                   --n_workers $NWORKERS \
                                   --wall_time $REMY_WALLTIME \
                                   --overwrite \

python queue_sleep_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 5 \
                                   --data_type 'clusterless' \
                                   --n_cores $NCORES \
                                   --n_workers $NWORKERS \
                                   --wall_time $REMY_WALLTIME \
                                   --overwrite \


# Sorted Spikes
python queue_sleep_cluster_jobs.py --data_type 'sorted_spikes' --n_cores $NCORES --n_workers $NWORKERS --wall_time $SORTED_WALLTIME
