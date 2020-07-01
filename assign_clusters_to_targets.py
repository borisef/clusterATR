import cluster_utils
import numpy as np
import pandas as pd

#will assign targets to clusters the best way possible
#each target is mixture of up to 3 targets

#inputs
targets_csv = "data/targets.csv" # from tiles
clusters_pickle = "results/clustered_without_false.pckl" # from cluster_points.py results

#params
WEIGHT_CLASS = 0.6
WEIGHT_COLOR = 1 - WEIGHT_CLASS
BONUS_TO_ANY_CLASS = 0.05
BONUS_TO_ANY_COLOR = 0.1
ASSIGN_ONE2ONE = True
ASSIGN_ONE2MANY = not ASSIGN_ONE2ONE
FAILURE_SIMILARITY_THRESHOLD = 0.2 # don't assign such similarity to no target


#load targets.csv
df_targets = pd.read_csv(targets_csv)

#load clustered.csv
df_clusters = pd.read_pickle(clusters_pickle)

#find distributions of targets's color and type
agg_df_targets =  cluster_utils.AggregateTargets(df_targets)

#compute scores (KL)
D = cluster_utils.ComputeDistancesTargets2Clusters(agg_df_targets, df_clusters, WEIGHT_CLASS, BONUS_TO_ANY_CLASS, BONUS_TO_ANY_COLOR)

print(np.round( 100.0 -  D*100.0, decimals=0))

if(ASSIGN_ONE2ONE):
    pass#solve assignment
else:
    pass# find max in every row


# add target column



print("OK")